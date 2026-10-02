"""Baseline for the deployment comparison: EEGNet-8,2 as implemented in this repository."""
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
