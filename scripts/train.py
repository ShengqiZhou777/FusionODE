from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import os
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
# Ensure project root is in sys.path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from src.data.io import read_cnn, read_morph, read_targets
from src.data.preprocess import build_trajectories
from src.data.dataset import SlidingWindowDataset
from src.data.collate import collate_windows

from src.models.full_model import (
    FusionGRUModel,
    FusionLSTMModel,
    FusionODERNNModel,
    StaticMLPBaseline,
)
from src.train.engine import train_one_epoch, eval_one_epoch
from src.utils.seed import set_seed


TARGET_NAMES = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]


def compute_target_scaler(ds_subset: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算 train 集 target 的 mean/std（按 4 维分别算）
    返回: y_mean [4], y_std [4]  (torch.float32)
    """
    ys = []
    for i in range(len(ds_subset)):
        item = ds_subset[i]
        ys.append(item["y"].float().unsqueeze(0))  # [1,4]
    Y = torch.cat(ys, dim=0)  # [N,4]
    y_mean = Y.mean(dim=0)
    y_std = Y.std(dim=0, unbiased=False)
    min_std = 0.05 * y_std.max()   # 也可以固定成 0.05 或 0.1
    y_std = y_std.clamp(min=min_std).clamp(min=1e-6)
    return y_mean, y_std


def build_run_dir(
    root: str,
    run_name: str | None,
    model_type: str,
    fusion_type: str,
    use_morph: bool,
    use_cnn: bool,
    target_condition: str,
    window_size: int,
) -> str:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mod = "morph" if use_morph else ""
        cnn = "cnn" if use_cnn else ""
        modality = "+".join([m for m in [mod, cnn] if m]) or "none"
        run_name = f"{timestamp}_{model_type}_{fusion_type}_{modality}_{target_condition}_w{window_size}"
    return os.path.join(root, "runs", run_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fusion ODE/RNN models with config files.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config entries, e.g. --set train.lr=1e-4 (repeatable).",
    )
    return parser.parse_args()


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required to read YAML configs.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_config(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yml", ".yaml"}:
        return _load_yaml(path)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config extension: {ext}")


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = config
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _apply_overrides(config: dict, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        _set_nested(config, key, _coerce_value(raw_value))


def _require(cfg: Mapping[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]


def _inspect_split_overlap(traj_train: dict, traj_val: dict, traj_test: dict) -> None:
    """
    Basic sanity checks for leakage or degenerate splits.
    We only compare cell IDs at the same condition/time to avoid false alarms.
    """
    for cond in traj_train.keys():
        train_ids = traj_train[cond]["cell_ids"]
        val_ids = traj_val.get(cond, {}).get("cell_ids", [])
        test_ids = traj_test.get(cond, {}).get("cell_ids", [])
        for idx, ids in enumerate(train_ids):
            if idx < len(val_ids):
                overlap = set(ids).intersection(val_ids[idx])
                if overlap:
                    print(f"[Warn] Train/Val leakage at {cond} index {idx}: {len(overlap)} shared cells")
            if idx < len(test_ids):
                overlap = set(ids).intersection(test_ids[idx])
                if overlap:
                    print(f"[Warn] Train/Test leakage at {cond} index {idx}: {len(overlap)} shared cells")


def _prepare_overfit_subset(ds_train: SlidingWindowDataset, ds_val: SlidingWindowDataset, overfit_n: int):
    """
    Optional overfit sanity check: train/val on the same tiny subset.
    Useful to verify the baseline can fit small data without logic bugs.
    """
    if overfit_n <= 0:
        return ds_train, ds_val
    subset_n = min(overfit_n, len(ds_train))
    subset_indices = list(range(subset_n))
    return Subset(ds_train, subset_indices), Subset(ds_train, subset_indices)


def _build_time_scale(traj_train: dict, target_condition: str, time_scale: float | None) -> float:
    if time_scale is not None:
        return time_scale
    if target_condition in traj_train:
        return float(traj_train[target_condition]["times"].max().item())
    return 1.0


def _eval_leave_one_timepoint(
    model: torch.nn.Module,
    dataset: SlidingWindowDataset,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    batch_size: int,
    target_names: list[str],
    run_dir: str,
    condition: str,
    window_size: int,
) -> None:
    """
    Leave-one-timepoint-out evaluation.
    With few timepoints, R2 can be unstable, so we also report MAE/RMSE.
    """
    time_to_indices: dict[float, list[int]] = {}
    for i in range(len(dataset)):
        t = float(dataset[i]["times"][-1])
        time_to_indices.setdefault(t, []).append(i)

    if not time_to_indices:
        print("[LOOTO] No timepoints found for evaluation.")
        return

    rows = []
    for time_value, indices in sorted(time_to_indices.items()):
        subset = Subset(dataset, indices)
        dl = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_windows,
        )
        metrics = eval_one_epoch(
            model,
            dl,
            device,
            y_mean=y_mean,
            y_std=y_std,
            target_names=target_names,
        )
        metrics_row = {"time": time_value, "n": len(indices)}
        metrics_row.update(metrics)
        rows.append(metrics_row)

    looto_path = os.path.join(run_dir, f"looto_metrics_{condition}_w{window_size}.csv")
    with open(looto_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[LOOTO] Saved metrics to {looto_path}")


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    _apply_overrides(config, args.set)

    train_cfg = _require(config, "train")
    model_cfg = _require(config, "model")
    data_cfg = _require(config, "data")
    eval_cfg = config.get("eval", {})
    run_cfg = config.get("run", {})
    set_seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    root = os.path.dirname(os.path.dirname(__file__))
    path_cnn = os.path.join(root, "data", "cnn_features_pca.csv")
    path_morph = os.path.join(root, "data", "morph_features.csv")
    path_tgt = os.path.join(root, "data", "result.csv")

    # ---- hyperparams ----
    window_size = train_cfg.get("window_size", 4)
    predict_last = True
    n_cells_per_bag = train_cfg.get("n_cells_per_bag", 500)
    batch_size = train_cfg.get("batch_size", 8)
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    epochs = train_cfg.get("epochs", 200)
    mc_test_runs = eval_cfg.get("mc_test_runs", 20)
    mc_test_base_seed = eval_cfg.get("mc_test_base_seed", 1000)

    # model ablations
    model_type = model_cfg.get("type", "odernn")  # "odernn", "gru", "lstm", "static"
    fusion_type = model_cfg.get("fusion_type", "cross_attention")  # "cross_attention" or "concat"
    use_morph = model_cfg.get("use_morph", True)
    use_cnn = model_cfg.get("use_cnn", True)

    # condition to train on
    target_condition = train_cfg.get("target_condition", "Light")

    # early stopping
    patience = train_cfg.get("patience", 15)
    min_delta = train_cfg.get("min_delta", 0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print(f"[Target Condition] {target_condition}")

    # ---- load + build trajectories (Cell-Level Split) ----
    df_cnn = read_cnn(data_cfg.get("cnn_path", path_cnn))
    df_morph = read_morph(data_cfg.get("morph_path", path_morph))
    df_tgt = read_targets(data_cfg.get("target_path", path_tgt))

    # 返回 (traj_train, traj_val, traj_test) (Contains ALL conditions)
    print(f"[Build] Splitting cells 70% Train / 15% Val / 15% Test per timepoint...")
    traj_train, traj_val, traj_test = build_trajectories(
        df_cnn, df_morph, df_tgt,
        n_cells_per_bag=n_cells_per_bag,
        split_ratios=(0.7, 0.15, 0.15),
        seed=0,
        split_seed=0,
        bag_seed=0,
        sample_with_replacement=True,
    )

    # ---- Filter for Single Condition ----
    print(f"[Filter] Keeping only condition: {target_condition}")
    traj_train = {k: v for k, v in traj_train.items() if k == target_condition}
    traj_val   = {k: v for k, v in traj_val.items()   if k == target_condition}
    traj_test  = {k: v for k, v in traj_test.items()  if k == target_condition}

    # ---- dataset ----
    ds_train = SlidingWindowDataset(traj_train, window_size=window_size, predict_last=predict_last)
    ds_val = SlidingWindowDataset(traj_val, window_size=window_size, predict_last=predict_last)
    ds_test = SlidingWindowDataset(traj_test, window_size=window_size, predict_last=predict_last)
    ds_train, ds_val = _prepare_overfit_subset(ds_train, ds_val, train_cfg.get("overfit_n", 0))
    # early stopping & scheduler
    patience = train_cfg.get("patience", 15)
    bad_epochs = 0
    best_val_norm = float("inf")
    best_val_raw = float("inf")
    
    # Scheduler params
    use_scheduler = train_cfg.get("use_scheduler", True)
    scheduler_factor = train_cfg.get("scheduler_factor", 0.5)
    scheduler_patience = train_cfg.get("scheduler_patience", 5)
    min_lr = train_cfg.get("min_lr", 1e-6)
    
    # Gradient clipping
    grad_clip = train_cfg.get("grad_clip", 1.0)

    print(f"[Dataset] train={len(ds_train)} | val={len(ds_val)} | test={len(ds_test)}")
    if not train_cfg.get("skip_inspect_data", False):
        _inspect_split_overlap(traj_train, traj_val, traj_test)

    # ---- target scaler from TRAIN only ----
    y_mean, y_std = compute_target_scaler(ds_train)
    print(f"[Target scaler]\n y_mean: {y_mean.tolist()}\n y_std : {y_std.tolist()}")

    # 保存 scaler
    run_dir = build_run_dir(
        root,
        run_cfg.get("name"),
        model_type,
        fusion_type,
        use_morph,
        use_cnn,
        target_condition,
        window_size,
    )
    os.makedirs(run_dir, exist_ok=True)
    scaler_path = os.path.join(run_dir, f"scaler_{target_condition}.pt")
    torch.save({"y_mean": y_mean, "y_std": y_std}, scaler_path)
    print("[Saved scaler]", scaler_path)

    # ---- loaders ----
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
        generator=g,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
        generator=g,
    )

    # ---- model ----
    batch0 = next(iter(dl_train))
    Dm = batch0["morph"].shape[-1]
    Dc = batch0["bags"].shape[-1]
    
    time_scale = _build_time_scale(traj_train, target_condition, model_cfg.get("time_scale"))
    time_features = model_cfg.get("time_features", "absolute+delta")

    if model_type == "odernn":
        model = FusionODERNNModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            ode_hidden=model_cfg.get("ode_hidden", 128),
            dropout=model_cfg.get("dropout", 0.1),
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
            attn_dim=model_cfg.get("attn_dim", 64),
            attn_heads=model_cfg.get("attn_heads", 4),
        ).to(device)
    elif model_type == "gru":
        model = FusionGRUModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            rnn_hidden=model_cfg.get("rnn_hidden", 128),
            rnn_layers=model_cfg.get("rnn_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
            attn_dim=model_cfg.get("attn_dim", 64),
            attn_heads=model_cfg.get("attn_heads", 4),
            use_time=time_features != "none",
            time_features=time_features,
            time_scale=time_scale,
        ).to(device)
    elif model_type == "lstm":
        model = FusionLSTMModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            rnn_hidden=model_cfg.get("rnn_hidden", 128),
            rnn_layers=model_cfg.get("rnn_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
            attn_dim=model_cfg.get("attn_dim", 64),
            attn_heads=model_cfg.get("attn_heads", 4),
            use_time=time_features != "none",
            time_features=time_features,
            time_scale=time_scale,
        ).to(device)
    elif model_type == "static":
        model = StaticMLPBaseline(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            dropout=model_cfg.get("dropout", 0.1),
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
            attn_dim=model_cfg.get("attn_dim", 64),
            attn_heads=model_cfg.get("attn_heads", 4),
            use_time=time_features != "none",
            time_features=time_features,
            time_scale=time_scale,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr
        )

    # ---- train loop ----
    ckpt_path = os.path.join(run_dir, f"best_{model_type}_{target_condition}.pt")

    target_names = TARGET_NAMES

    for ep in range(epochs):
        tr = train_one_epoch(model, dl_train, optimizer, device, y_mean=y_mean, y_std=y_std, grad_clip=grad_clip)
        va = eval_one_epoch(
            model,
            dl_val,
            device,
            y_mean=y_mean,
            y_std=y_std,
            target_names=target_names,
        )

        # Scheduler step
        if use_scheduler:
            scheduler.step(va["mse_norm"])
            
        cur_lr = optimizer.param_groups[0]["lr"]

        # Checkpoint (save based on Normalized MSE)
        if va["mse_norm"] < best_val_norm:
            best_val_norm = va["mse_norm"]
            best_val_raw = va["mse_raw"]
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "config": {}},
                ckpt_path,
            )
            # 只有在 improve 的时候打印详细指标，避免刷屏
            print(
                "  [New Best] "
                f"RMSE: Dry_Weight={va['rmse_raw_Dry_Weight']:.3f}, "
                f"Chl_Per_Cell={va['rmse_raw_Chl_Per_Cell']:.3f}, "
                f"Fv_Fm={va['rmse_raw_Fv_Fm']:.3f}, "
                f"Oxygen_Rate={va['rmse_raw_Oxygen_Rate']:.3f}"
            )
        else:
            bad_epochs += 1

        print(
            f"Epoch {ep:03d} | lr={cur_lr:.2e} | "
            f"Train Loss={tr['mse_norm']:.4f} | "
            f"Val Loss={va['mse_norm']:.4f} Val MSE={va['mse_raw']:.4f} | "
            f"bad_epochs={bad_epochs}/{patience}"
        )

        if bad_epochs >= patience:
            print(f"[EarlyStop] no improvement in val_mse_raw for {patience} epochs. Stop at epoch {ep}.")
            break

    print("[Done] best_val_mse_raw  =", best_val_raw)
    print("[Done] best_val_mse_norm =", best_val_norm)
    print("[Best ckpt]", ckpt_path)

    # ---- Final Test Evaluation ----
    print("\n[Test] Loading best checkpoint for final evaluation...")
    if len(ds_test) > 0:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

        if mc_test_runs <= 1:
            dl_test = DataLoader(
                ds_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_windows,
                generator=g,
            )
            test_metrics = eval_one_epoch(
                model,
                dl_test,
                device,
                y_mean=y_mean,
                y_std=y_std,
                target_names=target_names,
            )
            print(
                f"[Test Result] mse_raw={test_metrics['mse_raw']:.6f} | "
                f"mse_norm={test_metrics['mse_norm']:.6f}"
            )
            print(
                "[Test RMSE] "
                f"Dry_Weight={test_metrics['rmse_raw_Dry_Weight']:.4f}, "
                f"Chl_Per_Cell={test_metrics['rmse_raw_Chl_Per_Cell']:.4f}, "
                f"Fv_Fm={test_metrics['rmse_raw_Fv_Fm']:.4f}, "
                f"Oxygen_Rate={test_metrics['rmse_raw_Oxygen_Rate']:.4f}"
            )
            print(
                "[Test R2] "
                f"Dry_Weight={test_metrics['r2_Dry_Weight']:.4f}, "
                f"Chl_Per_Cell={test_metrics['r2_Chl_Per_Cell']:.4f}, "
                f"Fv_Fm={test_metrics['r2_Fv_Fm']:.4f}, "
                f"Oxygen_Rate={test_metrics['r2_Oxygen_Rate']:.4f}"
            )
        else:
            metrics_list = []
            all_preds_rows = []
            target_names = TARGET_NAMES

            for run_idx in range(mc_test_runs):
                traj_train_mc, traj_val_mc, traj_test_mc = build_trajectories(
                    df_cnn, df_morph, df_tgt,
                    n_cells_per_bag=n_cells_per_bag,
                    split_ratios=(0.7, 0.15, 0.15),
                    seed=0,
                    split_seed=0,
                    bag_seed=mc_test_base_seed + run_idx,
                    sample_with_replacement=True,
                )
                traj_test_mc = {k: v for k, v in traj_test_mc.items() if k == target_condition}
                if not traj_test_mc:
                    continue

                ds_test_mc = SlidingWindowDataset(traj_test_mc, window_size=window_size, predict_last=predict_last)
                dl_test_mc = DataLoader(
                    ds_test_mc,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_windows,
                    generator=g,
                )
                metrics, Y_true, Y_pred = eval_one_epoch(
                    model,
                    dl_test_mc,
                    device,
                    y_mean=y_mean,
                    y_std=y_std,
                    return_preds=True,
                    target_names=target_names,
                )
                metrics_list.append(metrics)

                # Collect predictions
                # Y_true, Y_pred: [N, 4]
                N = Y_true.shape[0]
                for i in range(N):
                    for t_idx, t_name in enumerate(target_names):
                        all_preds_rows.append({
                            "run": run_idx,
                            "target": t_name,
                            "true": float(Y_true[i, t_idx]),
                            "pred": float(Y_pred[i, t_idx])
                        })

            if not metrics_list:
                print("[MC Test] No valid test runs completed.")
                return

            def _mean_std(key: str) -> tuple[float, float]:
                vals = np.array([m[key] for m in metrics_list], dtype=np.float64)
                return float(vals.mean()), float(vals.std(ddof=0))

            mean_mse_raw, std_mse_raw = _mean_std("mse_raw")
            mean_mse_norm, std_mse_norm = _mean_std("mse_norm")
            
            mean_rmse_t0, std_rmse_t0 = _mean_std("rmse_raw_Dry_Weight")
            mean_rmse_t1, std_rmse_t1 = _mean_std("rmse_raw_Chl_Per_Cell")
            mean_rmse_t2, std_rmse_t2 = _mean_std("rmse_raw_Fv_Fm")
            mean_rmse_t3, std_rmse_t3 = _mean_std("rmse_raw_Oxygen_Rate")

            mean_r2_t0, std_r2_t0 = _mean_std("r2_Dry_Weight")
            mean_r2_t1, std_r2_t1 = _mean_std("r2_Chl_Per_Cell")
            mean_r2_t2, std_r2_t2 = _mean_std("r2_Fv_Fm")
            mean_r2_t3, std_r2_t3 = _mean_std("r2_Oxygen_Rate")

            print(f"[MC Test] runs={mc_test_runs}")
            print(
                f"[MC Result] mse_raw={mean_mse_raw:.6f}±{std_mse_raw:.6f} | "
                f"mse_norm={mean_mse_norm:.6f}±{std_mse_norm:.6f}"
            )
            print(
                "[MC RMSE] "
                f"Dry_Weight={mean_rmse_t0:.4f}±{std_rmse_t0:.4f}, "
                f"Chl_Per_Cell={mean_rmse_t1:.4f}±{std_rmse_t1:.4f}, "
                f"Fv_Fm={mean_rmse_t2:.4f}±{std_rmse_t2:.4f}, "
                f"Oxygen_Rate={mean_rmse_t3:.4f}±{std_rmse_t3:.4f}"
            )
            print(
                "[MC R2] "
                f"Dry_Weight={mean_r2_t0:.4f}±{std_r2_t0:.4f}, "
                f"Chl_Per_Cell={mean_r2_t1:.4f}±{std_r2_t1:.4f}, "
                f"Fv_Fm={mean_r2_t2:.4f}±{std_r2_t2:.4f}, "
                f"Oxygen_Rate={mean_r2_t3:.4f}±{std_r2_t3:.4f}"
            )

            # Save metrics CSV (existing)
            mc_path = os.path.join(
                run_dir,
                f"mc_metrics_{target_condition}_w{window_size}_k{mc_test_runs}.csv",
            )
            fieldnames = sorted(metrics_list[0].keys())
            with open(mc_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["run"] + fieldnames)
                writer.writeheader()
                for i, m in enumerate(metrics_list):
                    row = {"run": i}
                    row.update(m)
                    writer.writerow(row)
            print(f"[MC Metrics Saved] {mc_path}")

            # Save preds CSV (new)
            preds_path = os.path.join(
                run_dir,
                f"mc_preds_{target_condition}_w{window_size}_k{mc_test_runs}.csv",
            )
            with open(preds_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["run", "target", "true", "pred"])
                writer.writeheader()
                writer.writerows(all_preds_rows)
            print(f"[MC Preds Saved] {preds_path}")
    else:
        print("[Test] No test data available (ds_test is empty).")

    if eval_cfg.get("eval_looto", False) and len(ds_test) > 0:
        _eval_leave_one_timepoint(
            model,
            ds_test,
            device,
            y_mean=y_mean,
            y_std=y_std,
            batch_size=batch_size,
            target_names=target_names,
            run_dir=run_dir,
            condition=target_condition,
            window_size=window_size,
        )


if __name__ == "__main__":
    main()
