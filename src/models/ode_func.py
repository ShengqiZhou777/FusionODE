# src/models/ode_func.py
from __future__ import annotations
import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    MLP for ODE derivative: dh/dt = Net([h, u]) if control is provided.
    """
    def __init__(
        self,
        hidden_dim: int,
        ode_hidden: int = 128,
        dropout: float = 0.0,
        control_dim: int = 0,
    ):
        super().__init__()
        self.control_dim = control_dim
        in_dim = hidden_dim + control_dim if control_dim > 0 else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, ode_hidden),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ode_hidden, hidden_dim),
        )

    def forward(self, h: torch.Tensor, control: torch.Tensor | None = None) -> torch.Tensor:
        if self.control_dim > 0:
            if control is None:
                raise ValueError("control input required when control_dim > 0")
            h = torch.cat([h, control], dim=-1)
        return self.net(h)
