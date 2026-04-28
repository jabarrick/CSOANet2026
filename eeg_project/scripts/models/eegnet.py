"""EEGNet-8,2 — Lawhern et al., J. Neural Eng. 15 (2018) 056013."""
import torch, torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels=62, n_timepoints=2000, n_classes=2,
                 F1=8, D=2, F2=None, kern_length=251, sep_kern_length=125,
                 pool1=(1,4), pool2=(1,8), drop_rate=0.5):
        super().__init__()
        if F2 is None: F2 = F1 * D
        assert kern_length % 2 == 1 and sep_kern_length % 2 == 1
        self.n_channels, self.n_timepoints = n_channels, n_timepoints
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, (kern_length-1)//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(True), nn.AvgPool2d(pool1), nn.Dropout(drop_rate))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, sep_kern_length), padding=(0, (sep_kern_length-1)//2), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(True), nn.AvgPool2d(pool2), nn.Dropout(drop_rate))
        with torch.no_grad():
            flat = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_timepoints))).numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, n_classes))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.classifier(self.block2(self.block1(x)))
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
