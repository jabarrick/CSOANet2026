"""CSOANet2026-v2: a deployment-oriented variant of CSOANet2026 for int8 MCUs.

Same three stages and the same Cross-Scale Oscillatory Attention idea as v1
(csoanet2026.py); what changes is everything that made v1 expensive or
awkward on a Cortex-M class target:

  1. Spatial conv first.  v1 runs the temporal conv on all C raw channels
     (F1 x C x T/2 activations, ~94 % of the MACs).  v2 projects C channels to
     F2 virtual channels first, then runs a depthwise temporal conv on those.
     On a device the projection can be applied sample by sample as the ADC
     delivers data, so the raw C x T window never has to be buffered.
  2. Optional input decimation (`decim`), implemented as a stride of the
     spatial conv.  Only valid if the input is already band-limited below
     fs / (2 * decim), e.g. 1 kHz data low-passed at 40 Hz with decim=4.
  3. Bounded CSOA terms (`bound=True`).  Each branch output is batch-normed
     (folded away at export) and clipped to [-1, 1], so the three scale terms,
     the three coupling products and their convex combination all live in
     [-1, 1] and share one fixed-point format.  v1's unbounded products square
     the dynamic range, which is what hurts int8.
  4. Static gate (`gate="static"`).  Six learned logits instead of a per-trial
     Linear(6,6) on global means.  The fusion then needs no second pass over
     the trial and no stored product maps.  `gate="dynamic"` restores v1.
  5. ReLU instead of ELU (`act="relu"`): fuses into the preceding conv.
  6. The depth-attention vector `da` is dropped: a per-channel scale directly
     in front of a linear layer is redundant with that layer.

Every change is a constructor flag, so each one can be ablated on its own.
Kernel sizes come from v1's `auto_cfg`, evaluated on the decimated length, so
with decim=1 v1 and v2 share all temporal hyper-parameters.

    model = CSOANetV2(ch=64, t=640, nc=4)            # train
    deploy = model.fuse()                            # BN folded, ready to export
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

try:
    from .csoanet2026 import auto_cfg
except ImportError:  # run as a plain script
    from csoanet2026 import auto_cfg

TERM_NAMES = ("fast", "mid", "slow", "fast*mid", "mid*slow", "fast*slow")


def _act(name):
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"act must be 'relu' or 'elu', got {name!r}")


class CSOANetV2(nn.Module):
    """Input (B, 1, C, T) -> logits (B, nc)."""

    def __init__(self, ch, t, nc=2, decim=1, gate="static", bound=True, act="relu", **overrides):
        super().__init__()
        if gate not in ("static", "dynamic"):
            raise ValueError("gate must be 'static' or 'dynamic'")
        c = auto_cfg(ch, t // decim, nc)
        c.update(overrides)
        c.update(decim=decim, gate=gate, bound=bound, act=act, t_in=t)
        self.c = c
        F2, kt, k, dr = c["F1"] * c["D"], c["kt"], c["k"], c["dr"]

        # Stage 1: spatial projection, then depthwise temporal conv (halves T)
        self.spatial = nn.Conv2d(1, F2, (ch, 1), stride=(1, decim), bias=False)
        self.temporal = nn.Conv2d(F2, F2, (1, kt), stride=(1, 2), padding=(0, (kt - 1) // 2),
                                  groups=F2, bias=False)
        self.bn1 = nn.BatchNorm2d(F2)
        self.act1 = _act(act)
        self.pool1 = nn.AvgPool2d((1, c["p1"]))
        self.drop1 = nn.Dropout(dr)

        # Stage 2: CSOA - three dilated depthwise branches and their pairwise products
        def branch(dil):
            return nn.Conv2d(F2, F2, (1, k), dilation=(1, dil), padding=(0, dil * (k - 1) // 2),
                             groups=F2, bias=False)
        self.branches = nn.ModuleList([branch(1), branch(c["d1"]), branch(c["d2"])])
        self.branch_bn = nn.ModuleList([nn.BatchNorm2d(F2) if bound else nn.Identity() for _ in range(3)])
        self.clip = nn.Hardtanh(-1.0, 1.0) if bound else nn.Identity()
        if gate == "static":
            self.gate_logits = nn.Parameter(torch.zeros(6))
        else:
            self.gate = nn.Linear(6, 6)
        self.bn2 = nn.BatchNorm2d(F2)
        self.act2 = _act(act)

        # Stage 3: depthwise-separable conv and classifier
        self.dw3 = nn.Conv2d(F2, F2, (1, k), padding=(0, (k - 1) // 2), groups=F2, bias=False)
        self.pw3 = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act3 = _act(act)
        self.pool3 = nn.AvgPool2d((1, c["p2"]))
        self.drop3 = nn.Dropout(dr)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(F2, nc)

    # ------------------------------------------------------------------ forward
    def _terms(self, x):
        x0, x1, x2 = (self.clip(bn(conv(x))) for conv, bn in zip(self.branches, self.branch_bn))
        return [x0, x1, x2, x0 * x1, x1 * x2, x0 * x2]

    def _weights(self, terms):
        """(6,) for the static gate, (B, 6) for the dynamic one."""
        if self.c["gate"] == "static":
            return F.softmax(self.gate_logits, 0)
        g = torch.stack([t.mean((1, 2, 3)) for t in terms], 1)
        return F.softmax(self.gate(g), 1)

    def stage1(self, x):
        return self.drop1(self.pool1(self.act1(self.bn1(self.temporal(self.spatial(x))))))

    def forward(self, x):
        x = self.stage1(x)
        terms = self._terms(x)
        w = self._weights(terms)
        if w.dim() == 1:
            fused = sum(w[i] * terms[i] for i in range(6))
        else:
            fused = sum(w[:, i, None, None, None] * terms[i] for i in range(6))
        x = self.act2(self.bn2(fused + x))
        x = self.drop3(self.pool3(self.act3(self.bn3(self.pw3(self.dw3(x))))))
        return self.fc(self.gap(x).flatten(1))

    @torch.no_grad()
    def gate_weights(self, x):
        """Mean CSOA weights for a batch, in TERM_NAMES order (interpretability)."""
        w = self._weights(self._terms(self.stage1(x)))
        return w if w.dim() == 1 else w.mean(0)

    # ------------------------------------------------------------------ training helper
    @staticmethod
    def max_norm(model, v=0.25):
        """Same constraint as v1: L2 norm of every filter / row clipped to v."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "weight" in name and p.dim() >= 2:
                    nr = p.norm(2, dim=tuple(range(1, p.dim())), keepdim=True)
                    p.mul_(torch.clamp(nr, max=v) / (nr + 1e-8))

    # ------------------------------------------------------------------ export
    def fuse(self):
        """Return an eval-mode copy with BatchNorm folded and dropout removed.

        bn1, the branch BNs and bn3 fold into the conv in front of them.  bn2
        follows an addition, so it becomes a 1x1 depthwise conv (a per-channel
        scale and offset).  Outputs match the unfused eval-mode model.
        """
        m = copy.deepcopy(self).eval()
        m.temporal, m.bn1 = fuse_conv_bn_eval(m.temporal, m.bn1), nn.Identity()
        if m.c["bound"]:
            m.branches = nn.ModuleList(fuse_conv_bn_eval(cv, bn) for cv, bn in zip(m.branches, m.branch_bn))
            m.branch_bn = nn.ModuleList(nn.Identity() for _ in range(3))
        m.pw3, m.bn3 = fuse_conv_bn_eval(m.pw3, m.bn3), nn.Identity()

        bn, F2 = m.bn2, m.bn2.num_features
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        affine = nn.Conv2d(F2, F2, 1, groups=F2, bias=True)
        with torch.no_grad():
            affine.weight.copy_(scale.view(F2, 1, 1, 1))
            affine.bias.copy_(bn.bias - bn.running_mean * scale)
        m.bn2 = affine
        m.drop1, m.drop3 = nn.Identity(), nn.Identity()
        for p in m.parameters():
            p.requires_grad_(False)
        return m


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    x = torch.randn(4, 1, 64, 640)
    net = CSOANetV2(64, 640, nc=4).eval()
    print("cfg:", net.c)
    print("params:", n_params(net), "| deploy params:", n_params(net.fuse()))
    with torch.no_grad():
        print("out:", tuple(net(x).shape), "| fuse max abs diff:", float((net(x) - net.fuse()(x)).abs().max()))
