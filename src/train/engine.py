# # src/train/engine.py
# from __future__ import annotations
# from typing import Dict, Tuple
# import torch
# import torch.nn.functional as F


# def _to_device(batch: Dict, device: torch.device) -> Dict:
#     out = {}
#     for k, v in batch.items():
#         if torch.is_tensor(v):
#             out[k] = v.to(device, non_blocking=True)
#         else:
#             out[k] = v
#     return out


# @torch.no_grad()
# def eval_one_epoch(
#     model: torch.nn.Module,
#     dataloader,
#     device: torch.device,
# ) -> float:
#     model.eval()
#     losses = []
#     for batch in dataloader:
#         batch = _to_device(batch, device)
#         y_hat = model(batch)  # [B,4]
#         y = batch["y"]        # [B,4]
#         loss = F.mse_loss(y_hat, y)
#         losses.append(loss.detach().item())
#     return float(sum(losses) / max(len(losses), 1))


# def train_one_epoch(
#     model: torch.nn.Module,
#     dataloader,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     grad_clip: float | None = 1.0,
# ) -> float:
#     model.train()
#     losses = []
#     for batch in dataloader:
#         batch = _to_device(batch, device)

#         optimizer.zero_grad(set_to_none=True)
#         y_hat = model(batch)
#         y = batch["y"]
#         loss = F.mse_loss(y_hat, y)

#         loss.backward()
#         if grad_clip is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

#         optimizer.step()
#         losses.append(loss.detach().item())

#     return float(sum(losses) / max(len(losses), 1))
# src/train/engine.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Callable
import numpy as np
import torch

from src.train.losses import mse_loss
from src.train.metrics import batch_mse, summarize_regression_metrics


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    y_mean: Optional[torch.Tensor] = None,
    y_std: Optional[torch.Tensor] = None,
    return_preds: bool = False,
    target_names: Optional[list[str]] = None,
    batch_transform: Optional[Callable[[Dict], Dict]] = None,
) -> Dict[str, float] | tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Returns dict:
      - mse_norm: MSE in normalized target space (if y_mean/y_std provided), else equals mse_raw
      - mse_raw:  MSE in original target units
      - mae_raw_{target}: MAE for each target
      - rmse_raw_{target}: RMSE for each target
      - r2_{target}: R2 for each target
      
    If return_preds is True, returns (out_dict, Y_numpy, YH_numpy) instead.
    If batch_transform is provided, it is applied to the batch before moving to device.
    """
    model.eval()
    
    use_norm = (y_mean is not None) and (y_std is not None)
    if use_norm:
        y_mean = y_mean.to(device)
        y_std = y_std.to(device)

    all_y = []
    all_yhat = []

    mse_raw_list, mse_norm_list = [], []

    for batch in dataloader:
        if batch_transform is not None:
            batch = batch_transform(batch)
        batch = _to_device(batch, device)
        y_hat = model(batch)   # model 输出默认为 raw units: [B,4]
        y = batch["y"]

        all_y.append(y.detach().cpu())
        all_yhat.append(y_hat.detach().cpu())

        mse_raw, mse_norm = batch_mse(y_hat, y, y_mean=y_mean, y_std=y_std)
        mse_raw_list.append(mse_raw)
        if use_norm:
            mse_norm_list.append(mse_norm)

    if not all_y:
        if target_names is None:
            target_names = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
        print("[Warn] eval_one_epoch received an empty dataloader; returning NaN metrics.")
        empty_metrics: Dict[str, float] = {
            "mse_raw": float("inf"),
            "mse_norm": float("inf"),
        }
        for name in target_names:
            empty_metrics[f"mae_raw_{name}"] = float("nan")
            empty_metrics[f"rmse_raw_{name}"] = float("nan")
            empty_metrics[f"r2_{name}"] = float("nan")
        if return_preds:
            return empty_metrics, np.empty((0, 4)), np.empty((0, 4))
        return empty_metrics

    Y = torch.cat(all_y, dim=0)
    YH = torch.cat(all_yhat, dim=0)
    out = summarize_regression_metrics(
        Y,
        YH,
        mse_raw_list,
        mse_norm_list,
        use_norm,
        target_names=target_names,
    )
    
    if return_preds:
        return out, Y.numpy(), YH.numpy()
        
    return out


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    y_mean: Optional[torch.Tensor] = None,
    y_std: Optional[torch.Tensor] = None,
    grad_clip: float | None = 1.0,
    batch_transform: Optional[Callable[[Dict], Dict]] = None,
) -> Dict[str, float]:
    """
    Train with mse_norm if y_mean/y_std provided, else mse_raw.
    Returns dict with train losses:
      - mse_raw
      - mse_norm
    """
    model.train()
    mse_raw_list = []
    mse_norm_list = []

    use_norm = (y_mean is not None) and (y_std is not None)
    if use_norm:
        y_mean = y_mean.to(device)
        y_std = y_std.to(device)

    for batch in dataloader:
        if batch_transform is not None:
            batch = batch_transform(batch)
        batch = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(batch)
        y = batch["y"]

        loss, mse_raw, mse_norm = mse_loss(y_hat, y, y_mean=y_mean, y_std=y_std)
        mse_raw_list.append(mse_raw.detach().item())
        if use_norm:
            mse_norm_list.append(mse_norm.detach().item())

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    mse_raw = float(sum(mse_raw_list) / max(len(mse_raw_list), 1))
    if use_norm:
        mse_norm = float(sum(mse_norm_list) / max(len(mse_norm_list), 1))
    else:
        mse_norm = mse_raw

    return {"mse_raw": mse_raw, "mse_norm": mse_norm}
