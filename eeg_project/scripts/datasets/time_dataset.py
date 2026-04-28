"""Time-domain EEG dataset with per-epoch z-score normalization."""
from dataclasses import dataclass
import numpy as np, torch
from torch.utils.data import Dataset

@dataclass
class TimeConfig:
    tmin: float = -1.0; tmax: float = 1.0

class TimeDataset(Dataset):
    def __init__(self, epochs, labels, config=None):
        data = epochs.get_data()
        mean = data.mean(axis=2, keepdims=True)
        std = data.std(axis=2, keepdims=True) + 1e-8
        self.X = torch.from_numpy((data - mean) / std).float().unsqueeze(1)
        self.y = torch.from_numpy(labels).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
