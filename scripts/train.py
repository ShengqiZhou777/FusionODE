from __future__ import annotations
import os
import random
import csv
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader, Subset
# Ensure project root is in sys.path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from src.data.io import read_cnn, read_morph, read_targets
from src.data.preprocess import build_trajectories
from src.data.dataset import SlidingWindowDataset
from src.data.collate import collate_windows

from src.models.full_model import FusionODERNNModel
from src.train.engine import train_one_epoch, eval_one_epoch
from src.utils.seed import set_seed


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


def main():
    set_seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    root = os.path.dirname(os.path.dirname(__file__))
    path_cnn = os.path.join(root, "data", "cnn_features_pca.csv")
    path_morph = os.path.join(root, "data", "morph_features.csv")
    path_tgt = os.path.join(root, "data", "result.csv")

    # ---- hyperparams ----
    window_size = 5
    predict_last = True
    n_cells_per_bag = 500
    batch_size = 8
    lr = 1e-2
    weight_decay = 1e-4
    epochs = 200
    mc_test_runs = 1
    mc_test_base_seed = 1000

    # condition to train on
    target_condition = "Light"

    # early stopping
    patience = 15
    min_delta = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print(f"[Target Condition] {target_condition}")

    # ---- load + build trajectories (Cell-Level Split) ----
    df_cnn = read_cnn(path_cnn)
    df_morph = read_morph(path_morph)
    df_tgt = read_targets(path_tgt)

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

    print(f"[Dataset] train={len(ds_train)} | val={len(ds_val)} | test={len(ds_test)}")

    # ---- target scaler from TRAIN only ----
    y_mean, y_std = compute_target_scaler(ds_train)
    print("[Target scaler]")
    print(" y_mean:", y_mean.tolist())
    print(" y_std :", y_std.tolist())

    # 保存 scaler
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)
    scaler_path = os.path.join(root, "runs", f"scaler_{target_condition}.pt")
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

    model = FusionODERNNModel(
        morph_dim=Dm,
        cnn_dim=Dc,
        z_morph=64,
        z_cnn=64,
        hidden_dim=128,
        ode_hidden=128,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ---- training ----
    best_val_raw = float("inf")
    best_val_norm = float("inf")
    bad_epochs = 0

    ckpt_path = os.path.join(root, "runs", f"best_ode_{target_condition}.pt")

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, dl_train, optimizer, device, y_mean=y_mean, y_std=y_std, grad_clip=1.0)
        va = eval_one_epoch(model, dl_val, device, y_mean=y_mean, y_std=y_std)

        val_score = va["mse_raw"]                 # ✅ early stop 用 raw
        improved = (best_val_raw - val_score) > min_delta

        if improved:
            best_val_raw = val_score
            best_val_norm = va["mse_norm"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": ep,
                    "best_val_mse_raw": best_val_raw,
                    "best_val_mse_norm": best_val_norm,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "target_condition": target_condition,
                },
                ckpt_path,
            )
            # 只有在 improve 的时候打印详细指标，避免刷屏
            print(
                "  [New Best] "
                f"val_rmse: t0={va['rmse_raw_t0']:.4f}, t1={va['rmse_raw_t1']:.4f}, "
                f"t2={va['rmse_raw_t2']:.4f}, t3={va['rmse_raw_t3']:.4f} | "
                f"val_r2_mean={va['r2_mean']:.4f}"
            )
        else:
            bad_epochs += 1

        print(
            f"Epoch {ep:03d} | "
            f"train_mse_norm={tr['mse_norm']:.6f} train_mse_raw={tr['mse_raw']:.6f} | "
            f"val_mse_norm={va['mse_norm']:.6f} val_mse_raw={va['mse_raw']:.6f} val_r2_mean={va['r2_mean']:.4f} | "
            f"best_val_raw={best_val_raw:.6f} best_val_norm={best_val_norm:.6f} | "
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
            test_metrics = eval_one_epoch(model, dl_test, device, y_mean=y_mean, y_std=y_std)
            print(
                f"[Test Result] mse_raw={test_metrics['mse_raw']:.6f} | "
                f"mse_norm={test_metrics['mse_norm']:.6f} | "
                f"r2_mean={test_metrics['r2_mean']:.4f}"
            )
            print(
                "[Test RMSE] "
                f"t0={test_metrics['rmse_raw_t0']:.4f}, t1={test_metrics['rmse_raw_t1']:.4f}, "
                f"t2={test_metrics['rmse_raw_t2']:.4f}, t3={test_metrics['rmse_raw_t3']:.4f}"
            )
            print(
                "[Test R2] "
                f"t0={test_metrics['r2_t0']:.4f}, t1={test_metrics['r2_t1']:.4f}, "
                f"t2={test_metrics['r2_t2']:.4f}, t3={test_metrics['r2_t3']:.4f}"
            )
        else:
            metrics_list = []
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
                ds_test_mc = SlidingWindowDataset(traj_test_mc, window_size=window_size, predict_last=predict_last)
                dl_test_mc = DataLoader(
                    ds_test_mc,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_windows,
                    generator=g,
                )
                metrics = eval_one_epoch(model, dl_test_mc, device, y_mean=y_mean, y_std=y_std)
                metrics_list.append(metrics)

            def _mean_std(key: str) -> tuple[float, float]:
                vals = np.array([m[key] for m in metrics_list], dtype=np.float64)
                return float(vals.mean()), float(vals.std(ddof=0))

            mean_mse_raw, std_mse_raw = _mean_std("mse_raw")
            mean_mse_norm, std_mse_norm = _mean_std("mse_norm")
            mean_r2, std_r2 = _mean_std("r2_mean")
            mean_rmse_t0, std_rmse_t0 = _mean_std("rmse_raw_t0")
            mean_rmse_t1, std_rmse_t1 = _mean_std("rmse_raw_t1")
            mean_rmse_t2, std_rmse_t2 = _mean_std("rmse_raw_t2")
            mean_rmse_t3, std_rmse_t3 = _mean_std("rmse_raw_t3")

            print(f"[MC Test] runs={mc_test_runs}")
            print(
                f"[MC Result] mse_raw={mean_mse_raw:.6f}±{std_mse_raw:.6f} | "
                f"mse_norm={mean_mse_norm:.6f}±{std_mse_norm:.6f} | "
                f"r2_mean={mean_r2:.4f}±{std_r2:.4f}"
            )
            print(
                "[MC RMSE] "
                f"t0={mean_rmse_t0:.4f}±{std_rmse_t0:.4f}, "
                f"t1={mean_rmse_t1:.4f}±{std_rmse_t1:.4f}, "
                f"t2={mean_rmse_t2:.4f}±{std_rmse_t2:.4f}, "
                f"t3={mean_rmse_t3:.4f}±{std_rmse_t3:.4f}"
            )

            os.makedirs(os.path.join(root, "runs"), exist_ok=True)
            mc_path = os.path.join(
                root,
                "runs",
                f"mc_test_{target_condition}_w{window_size}_k{mc_test_runs}.csv",
            )
            fieldnames = sorted(metrics_list[0].keys())
            with open(mc_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["run"] + fieldnames)
                writer.writeheader()
                for i, m in enumerate(metrics_list):
                    row = {"run": i}
                    row.update(m)
                    writer.writerow(row)
            print(f"[MC Saved] {mc_path}")
    else:
        print("[Test] No test data available (ds_test is empty).")


if __name__ == "__main__":
    main()
