# src/models/decoder.py
from __future__ import annotations
import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 4, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, H]
        return self.net(h)
