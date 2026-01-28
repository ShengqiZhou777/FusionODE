from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from src.train.losses import normalize_targets


def batch_mse(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    y_mean: Optional[torch.Tensor] = None,
    y_std: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    mse_raw = torch.mean((y_hat - y) ** 2).item()
    if y_mean is not None and y_std is not None:
        y_hat_n = normalize_targets(y_hat, y_mean, y_std)
        y_n = normalize_targets(y, y_mean, y_std)
        mse_norm = torch.mean((y_hat_n - y_n) ** 2).item()
    else:
        mse_norm = mse_raw
    return mse_raw, mse_norm


def summarize_regression_metrics(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    mse_raw_list: List[float],
    mse_norm_list: List[float],
    use_norm: bool,
    target_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    mae = (y_hat - y).abs().mean(dim=0)                 # [4]
    rmse = torch.sqrt(((y_hat - y) ** 2).mean(dim=0))   # [4]

    ss_res = ((y_hat - y) ** 2).sum(dim=0)
    y_mean = y.mean(dim=0)
    ss_tot = ((y - y_mean) ** 2).sum(dim=0)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))  # [4]

    mse_raw = float(sum(mse_raw_list) / max(len(mse_raw_list), 1))
    if use_norm:
        mse_norm = float(sum(mse_norm_list) / max(len(mse_norm_list), 1))
    else:
        mse_norm = mse_raw

    if target_names is None:
        target_names = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]

    metrics = {
        "mse_raw": mse_raw,
        "mse_norm": mse_norm,
    }
    for idx, name in enumerate(target_names):
        metrics[f"mae_raw_{name}"] = float(mae[idx])
        metrics[f"rmse_raw_{name}"] = float(rmse[idx])
        metrics[f"r2_{name}"] = float(r2[idx])
    return metrics
