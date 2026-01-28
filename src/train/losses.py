from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def normalize_targets(
    y: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    return (y - y_mean) / y_std


def mse_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    y_mean: Optional[torch.Tensor] = None,
    y_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse_raw = F.mse_loss(y_hat, y)
    if y_mean is not None and y_std is not None:
        y_hat_n = normalize_targets(y_hat, y_mean, y_std)
        y_n = normalize_targets(y, y_mean, y_std)
        mse_norm = F.mse_loss(y_hat_n, y_n)
    else:
        mse_norm = mse_raw
    loss = mse_norm
    return loss, mse_raw, mse_norm
