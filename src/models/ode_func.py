# src/models/ode_func.py
from __future__ import annotations
import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Standard MLP for defining the ODE derivative: dh/dt = Net(h)
    """
    def __init__(self, hidden_dim: int, ode_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ode_hidden),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ode_hidden, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
