"""ShallowConvNet & DeepConvNet — Schirrmeister et al., Human Brain Mapping 38 (2017).

Standard EEG-DL baselines alongside EEGNet. Adapted for variable input shapes.
"""
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────
# ShallowConvNet: band-power feature extractor (FBCSP-like)
# ─────────────────────────────────────────────────────────
class ShallowConvNet(nn.Module):
    """ShallowConvNet extracts log band-power features via temporal filtering,
    spatial filtering, squaring, log, and average pooling.

    Input:  (B, 1, C, T)
    Output: (B, n_classes)
    """
    def __init__(self, n_channels=62, n_timepoints=2000, n_classes=2,
                 n_filters=40, kern_length=25, pool_size=75, pool_stride=15,
                 drop_rate=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, kern_length), bias=False)
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.drop = nn.Dropout(drop_rate)

        # Compute flat features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.temporal(dummy)
            x = self.spatial(x)
            x = self.bn(x)
            x = x ** 2                      # square nonlinearity
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            flat = x.numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(drop_rate),
                                        nn.Linear(flat, n_classes))

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────
# DeepConvNet: generic deep architecture for EEG
# ─────────────────────────────────────────────────────────
class DeepConvNet(nn.Module):
    """DeepConvNet with 4 conv blocks. More generic than ShallowConvNet,
    captures hierarchical temporal features.

    Input:  (B, 1, C, T)
    Output: (B, n_classes)
    """
    def __init__(self, n_channels=62, n_timepoints=2000, n_classes=2,
                 n_filters=25, kern_length=10, drop_rate=0.5):
        super().__init__()
        # Block 1: temporal + spatial
        self.block1 = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, kern_length), bias=False),
            nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(drop_rate),
        )
        # Blocks 2-4: temporal only
        self.block2 = self._make_block(n_filters, n_filters * 2, kern_length, drop_rate)
        self.block3 = self._make_block(n_filters * 2, n_filters * 4, kern_length, drop_rate)
        self.block4 = self._make_block(n_filters * 4, n_filters * 8, kern_length, drop_rate)

        # Compute flat features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.block4(self.block3(self.block2(self.block1(dummy))))
            flat = x.numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, n_classes))
        self._init_weights()

    @staticmethod
    def _make_block(in_ch, out_ch, kern, drop):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, kern), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(drop),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
