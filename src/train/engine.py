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
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _norm_y(y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    # y: [B,4]
    return (y - y_mean) / y_std


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    y_mean: Optional[torch.Tensor] = None,
    y_std: Optional[torch.Tensor] = None,
    return_preds: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Returns dict:
      - mse_norm: MSE in normalized target space (if y_mean/y_std provided), else equals mse_raw
      - mse_raw:  MSE in original target units
      - mae_raw_t*: MAE for each target
      - rmse_raw_t*: RMSE for each target
      - r2_t*: R2 for each target
      
    If return_preds is True, returns (out_dict, Y_numpy, YH_numpy) instead.
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
        batch = _to_device(batch, device)
        y_hat = model(batch)   # model 输出默认为 raw units: [B,4]
        y = batch["y"]

        all_y.append(y.detach().cpu())
        all_yhat.append(y_hat.detach().cpu())

        mse_raw_list.append(F.mse_loss(y_hat, y).item())

        if use_norm:
            y_hat_n = _norm_y(y_hat, y_mean, y_std)
            y_n = _norm_y(y, y_mean, y_std)
            mse_norm = F.mse_loss(y_hat_n, y_n).item()
            mse_norm_list.append(mse_norm)

    Y = torch.cat(all_y, dim=0)       # [N,4]
    YH = torch.cat(all_yhat, dim=0)   # [N,4]

    mae = (YH - Y).abs().mean(dim=0)                 # [4]
    rmse = torch.sqrt(((YH - Y) ** 2).mean(dim=0))   # [4]

    # R2 Calculation per target
    ss_res = ((YH - Y) ** 2).sum(dim=0)
    y_mean = Y.mean(dim=0)
    ss_tot = ((Y - y_mean) ** 2).sum(dim=0)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) # [4]

    mse_raw = float(sum(mse_raw_list) / max(len(mse_raw_list), 1))
    if use_norm:
        mse_norm = float(sum(mse_norm_list) / max(len(mse_norm_list), 1))
    else:
        mse_norm = mse_raw
    out = {
        "mse_raw": mse_raw,
        "mse_norm": mse_norm,
        "mae_raw_t0": float(mae[0]), "mae_raw_t1": float(mae[1]), "mae_raw_t2": float(mae[2]), "mae_raw_t3": float(mae[3]),
        "rmse_raw_t0": float(rmse[0]), "rmse_raw_t1": float(rmse[1]), "rmse_raw_t2": float(rmse[2]), "rmse_raw_t3": float(rmse[3]),
        "r2_t0": float(r2[0]), "r2_t1": float(r2[1]), "r2_t2": float(r2[2]), "r2_t3": float(r2[3]),
    }
    
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
        batch = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(batch)
        y = batch["y"]

        mse_raw = F.mse_loss(y_hat, y)
        mse_raw_list.append(mse_raw.detach().item())

        if use_norm:
            y_hat_n = _norm_y(y_hat, y_mean, y_std)
            y_n = _norm_y(y, y_mean, y_std)
            loss = F.mse_loss(y_hat_n, y_n)
            mse_norm_list.append(loss.detach().item())
        else:
            loss = mse_raw

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
