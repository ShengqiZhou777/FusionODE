# src/models/odernn.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from src.models.ode_func import ODEFunc


def rk4_step(h: torch.Tensor, dt: torch.Tensor, f) -> torch.Tensor:
    """One RK4 step for dh/dt=f(h). dt: [B,1]"""
    k1 = f(h)
    k2 = f(h + 0.5 * dt * k1)
    k3 = f(h + 0.5 * dt * k2)
    k4 = f(h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_integrate(h: torch.Tensor, dt: torch.Tensor, f, dt_max: float = 0.05) -> torch.Tensor:
    """
    Integrate with multiple RK4 sub-steps so that each sub-step <= dt_max.
    dt: [B] (already normalized time scale, e.g., hours/72)
    """
    B, H = h.shape
    dt = torch.clamp(dt, min=1e-6)  # avoid 0
    # number of substeps per sample: ceil(dt/dt_max)
    n_steps = torch.ceil(dt / dt_max).to(torch.int64)  # [B]
    n_steps = torch.clamp(n_steps, min=1, max=1000)

    # We run max_steps and mask updates for samples that finished.
    max_steps = int(n_steps.max().item())
    dt_sub = (dt / n_steps).unsqueeze(-1)  # [B,1]

    for s in range(max_steps):
        active = (n_steps > s).unsqueeze(-1)  # [B,1] bool
        if not active.any():
            break
        h_new = rk4_step(h, dt_sub, f)
        # only update active samples
        h = torch.where(active, h_new, h)

    return h


class ODERNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        ode_hidden: int = 128,
        dropout: float = 0.0,
        dt_max: float = 0.05,   # <=0.05 works well when time normalized to [0,1]
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt_max = dt_max
        self.func = ODEFunc(hidden_dim=hidden_dim, ode_hidden=ode_hidden, dropout=dropout)
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B, W, Dx = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)

        for i in range(W):
            if i > 0:
                dt = times[:, i] - times[:, i - 1]  # [B]
                dt = torch.clamp(dt, min=0.0)
                h = rk4_integrate(h, dt, self.func, dt_max=self.dt_max)
            h = self.gru(x[:, i, :], h)

        return h
