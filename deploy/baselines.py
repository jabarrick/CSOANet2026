"""Baseline for the deployment comparison: EEGNet-8,2 as implemented in this repository."""
import copy

import torch
import torch.nn as nn

from eeg_project.scripts.models.eegnet import EEGNet


def _odd(n):
    return int(n) | 1


def make_eegnet(ch, t, nc, fs=160.0, drop=0.25):
    """EEGNet-8,2 (Lawhern et al. 2018).

    Kernels follow the paper's rule: temporal kernel of fs/2 samples, separable
    kernel of 0.5 s at the pooled rate (fs/8 samples).  Dropout 0.25 is the
    paper's value for cross-subject training.  fs=None keeps the class defaults,
    i.e. the 1 kHz configuration used in this repository's own experiments.
    """
    if fs is None:
        m = EEGNet(n_channels=ch, n_timepoints=t, n_classes=nc)
    else:
        m = EEGNet(n_channels=ch, n_timepoints=t, n_classes=nc, kern_length=_odd(fs / 2),
                   sep_kern_length=_odd(fs / 8), drop_rate=drop)
    # eegnet.py writes nn.ELU(True), which sets alpha=True rather than inplace.  Numerically that
    # is the standard ELU, but the boolean alpha breaks ONNX export, so use plain ELU modules.
    for block in (m.block1, m.block2):
        for i, layer in enumerate(block):
            if isinstance(layer, nn.ELU):
                block[i] = nn.ELU()
    return m


@torch.no_grad()
def eegnet_max_norm(model):
    """EEGNet's own constraints: max-norm 1 on the spatial filters, 0.25 on the classifier."""
    w = model.block1[2].weight
    w.copy_(torch.renorm(w, 2, 0, 1.0))
    w = model.classifier[1].weight
    w.copy_(torch.renorm(w, 2, 0, 0.25))


class EEGNetSpatialFirst(nn.Module):
    """A trained EEGNet rewritten with the spatial conv in front of the temporal conv.

    In EEGNet the temporal conv and the depthwise spatial conv are separated only by
    BatchNorm, which is affine at inference, so the two linear maps commute.  The
    rewritten network computes exactly the same function (up to float rounding) but
    its temporal conv runs on F1*D virtual channels instead of all C electrodes.
    This is change 1 of CSOANet-v2 applied to EEGNet, with no retraining.

    Build it with `from_module(trained_eegnet)` or `from_onnx(exported_fp32_file)`.
    """

    def __init__(self, Wt, bt, Ws, bs, pad_t, rest1, block2, classifier):
        """Wt, bt / Ws, bs: temporal and spatial conv with their BatchNorm already folded in."""
        super().__init__()
        F1, F2, kt = Wt.shape[0], Ws.shape[0], Wt.shape[-1]
        group = torch.arange(F2) // (F2 // F1)             # temporal filter feeding each spatial filter
        self.spatial = nn.Conv2d(1, F2, tuple(Ws.shape[2:]), bias=False)
        self.temporal = nn.Conv2d(F2, F2, (1, kt), padding=(0, pad_t), groups=F2, bias=True)
        with torch.no_grad():
            self.spatial.weight.copy_(Ws)
            self.temporal.weight.copy_(Wt[group])
            self.temporal.bias.copy_(bt[group] * Ws.sum(dim=(1, 2, 3)) + bs)
        self.rest1, self.block2, self.classifier = rest1, block2, classifier
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return self.classifier(self.block2(self.rest1(self.temporal(self.spatial(x)))))

    @classmethod
    def from_module(cls, eegnet):
        m = copy.deepcopy(eegnet).eval()
        conv_t, bn1, conv_s, bn2 = m.block1[0], m.block1[1], m.block1[2], m.block1[3]
        with torch.no_grad():
            a1 = bn1.weight / torch.sqrt(bn1.running_var + bn1.eps)
            a2 = bn2.weight / torch.sqrt(bn2.running_var + bn2.eps)
            Wt, bt = conv_t.weight * a1.view(-1, 1, 1, 1), bn1.bias - bn1.running_mean * a1
            Ws, bs = conv_s.weight * a2.view(-1, 1, 1, 1), bn2.bias - bn2.running_mean * a2
        return cls(Wt, bt, Ws, bs, conv_t.padding[1], nn.Sequential(*list(m.block1)[4:]), m.block2, m.classifier)

    @classmethod
    def from_onnx(cls, path):
        """Rebuild from the float32 ONNX file that quantize.export_fp32 wrote for a trained EEGNet."""
        import onnx
        from onnx import numpy_helper
        g = onnx.load(str(path)).graph
        t = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in g.initializer}
        for n in g.node:
            if n.op_type == "Identity" and n.input[0] in t:
                t[n.output[0]] = t[n.input[0]]
        convs = [n for n in g.node if n.op_type == "Conv"]
        pools = [n for n in g.node if n.op_type == "AveragePool"]
        gemm = [n for n in g.node if n.op_type == "Gemm"]
        if (len(convs), len(pools), len(gemm)) != (4, 2, 1):
            raise ValueError(f"{path} does not look like an exported EEGNet")

        def attr(n, name):
            return next(list(a.ints) for a in n.attribute if a.name == name)

        def conv(n, groups):
            w = t[n.input[1]]
            b = t[n.input[2]] if len(n.input) > 2 else None
            c = nn.Conv2d(w.shape[1] * groups, w.shape[0], tuple(w.shape[2:]), padding=(0, attr(n, "pads")[1]),
                          groups=groups, bias=b is not None)
            with torch.no_grad():
                c.weight.copy_(w)
                if b is not None:
                    c.bias.copy_(b)
            return c

        Wt, bt, Ws, bs = t[convs[0].input[1]], t[convs[0].input[2]], t[convs[1].input[1]], t[convs[1].input[2]]
        F2 = Ws.shape[0]
        fc = nn.Linear(t[gemm[0].input[1]].shape[1], t[gemm[0].input[1]].shape[0])
        with torch.no_grad():
            fc.weight.copy_(t[gemm[0].input[1]])
            fc.bias.copy_(t[gemm[0].input[2]])
        rest1 = nn.Sequential(nn.ELU(), nn.AvgPool2d(tuple(attr(pools[0], "kernel_shape"))))
        block2 = nn.Sequential(conv(convs[2], F2), conv(convs[3], 1), nn.ELU(),
                               nn.AvgPool2d(tuple(attr(pools[1], "kernel_shape"))))
        return cls(Wt, bt, Ws, bs, attr(convs[0], "pads")[1], rest1, block2, nn.Sequential(nn.Flatten(), fc))
