# src/models/time_aware_rnn.py
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.decoder import RegressionHead


def _build_time_feature(times: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Build time feature tensor of shape [B, T, 1].
    mode:
      - "delta": use dt_k = t_k - t_{k-1}, dt_0 = 0
      - "absolute": use t_k
    """
    if mode == "delta":
        dt = torch.zeros_like(times)
        dt[:, 1:] = times[:, 1:] - times[:, :-1]
        return dt.unsqueeze(-1)
    if mode == "absolute":
        return times.unsqueeze(-1)
    raise ValueError(f"Unsupported time feature mode: {mode}")


class TimeAwareGRURegressor(nn.Module):
    """
    Minimal time-aware GRU baseline for irregularly sampled sequences.

    Inputs:
      x: [B, T, D]
      times: [B, T]
    Outputs:
      y_hat: [B, 4]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        time_mode: str = "delta",
        use_time: bool = True,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.use_time = use_time
        self.time_mode = time_mode
        time_dim = 1 if use_time else 0
        self.gru = nn.GRU(
            input_size=input_dim + time_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = RegressionHead(
            in_dim=hidden_dim,
            out_dim=4,
            hidden=head_hidden,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if self.use_time:
            t_feat = _build_time_feature(times, self.time_mode)
            x = torch.cat([x, t_feat], dim=-1)
        _, h_last = self.gru(x)
        h = h_last[-1]
        return self.head(h)


class TimeAwareLSTMRegressor(nn.Module):
    """
    Minimal time-aware LSTM baseline for irregularly sampled sequences.

    Inputs:
      x: [B, T, D]
      times: [B, T]
    Outputs:
      y_hat: [B, 4]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        time_mode: str = "delta",
        use_time: bool = True,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.use_time = use_time
        self.time_mode = time_mode
        time_dim = 1 if use_time else 0
        self.lstm = nn.LSTM(
            input_size=input_dim + time_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = RegressionHead(
            in_dim=hidden_dim,
            out_dim=4,
            hidden=head_hidden,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if self.use_time:
            t_feat = _build_time_feature(times, self.time_mode)
            x = torch.cat([x, t_feat], dim=-1)
        _, (h_last, _) = self.lstm(x)
        h = h_last[-1]
        return self.head(h)
