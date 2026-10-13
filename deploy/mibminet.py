"""MI-BMInet (Wang, Hersche, Magno, Benini; IEEE Sensors Journal 24(6), 2024), re-implemented.

Follows the paper and the authors' reference code (github.com/pulp-platform/MI-BMInet,
Apache-2.0, where the model is still called edgeEEGNet).  It is a spatial-first network
designed for MCUs:

    spatial depthwise conv (Nk kernels, C x 1) -> BN
    temporal depthwise conv (1 x Nf, "same")   -> BN -> ReLU -> AvgPool 1x8 -> Dropout
    separable conv (1 x 16 depthwise, 1 x 1)   -> BN -> ReLU -> AvgPool 1x8 -> time-dropout
    dense

Paper settings: Nk = 16, Nf = 128 at 160 Hz (PhysioNet MM/MI); Nk = 32, Nf = 64 at 250 Hz
(BCI IV 2a).  With the paper's 3 s windows these give its parameter counts (4,228 / 6,084,
counting BatchNorm as 4 values per channel); `paper_param_count` checks that.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PAPER_SETTINGS = {160.0: dict(nk=16, nf=128), 250.0: dict(nk=32, nf=64)}


class TimeDropout2d(nn.Dropout2d):
    """Drops whole time steps across all feature maps (the authors' TimeDropout2d)."""

    def forward(self, x):
        if not self.training:
            return x
        return F.dropout2d(x.permute(0, 3, 1, 2), self.p, True, self.inplace).permute(0, 2, 3, 1)


class MIBMInet(nn.Module):
    """Input (B, 1, C, T) -> logits (B, nc)."""

    def __init__(self, ch, t, nc, fs=160.0, nk=None, nf=None, p_dropout=0.2):
        super().__init__()
        cfg = PAPER_SETTINGS.get(float(fs), PAPER_SETTINGS[160.0])
        nk, nf = nk or cfg["nk"], nf or cfg["nf"]
        self.c = dict(ch=ch, t=t, nc=nc, fs=fs, nk=nk, nf=nf, p_dropout=p_dropout)
        bn = dict(momentum=0.01, eps=1e-3)                     # Keras defaults, as in the reference code
        self.spatial = nn.Conv2d(1, nk, (ch, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(nk, **bn)
        self.pad_t = nn.ZeroPad2d((nf // 2 - 1, nf // 2, 0, 0))
        self.temporal = nn.Conv2d(nk, nk, (1, nf), groups=nk, bias=False)
        self.bn2 = nn.BatchNorm2d(nk, **bn)
        self.pool1 = nn.AvgPool2d((1, 8))
        self.drop1 = nn.Dropout(p_dropout)
        self.pad_s = nn.ZeroPad2d((7, 8, 0, 0))
        self.sep_dw = nn.Conv2d(nk, nk, (1, 16), groups=nk, bias=False)
        self.sep_pw = nn.Conv2d(nk, nk, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nk, **bn)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = TimeDropout2d(p_dropout)
        self.fc = nn.Linear(nk * ((t // 8) // 8), nc)
        for m in self.modules():                               # reference code: Xavier-uniform, zero bias
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn1(self.spatial(x))
        x = self.drop1(self.pool1(F.relu(self.bn2(self.temporal(self.pad_t(x))))))
        x = self.sep_pw(self.sep_dw(self.pad_s(x)))
        x = self.drop2(self.pool2(F.relu(self.bn3(x))))
        return self.fc(x.flatten(1))


def paper_param_count(model):
    """Parameter count as the paper reports it: weights without conv biases, BN as 4 values per channel."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            n += 4 * m.num_features
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            n += sum(p.numel() for p in m.parameters())
    return n
