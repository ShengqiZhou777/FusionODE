from __future__ import annotations
import argparse
import os
import random
import csv
from datetime import datetime
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

from src.models.full_model import FusionGRUModel, FusionLSTMModel, FusionODERNNModel
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
    parser = argparse.ArgumentParser(description="Train Fusion ODE/RNN models with ablations.")
    parser.add_argument("--model-type", choices=["odernn", "gru", "lstm"], default="odernn")
    parser.add_argument("--fusion-type", choices=["cross_attention", "concat"], default="cross_attention")
    parser.add_argument("--use-morph", action="store_true", default=True)
    parser.add_argument("--morph-only", dest="use_cnn", action="store_false")
    parser.add_argument("--use-cnn", action="store_true", default=True)
    parser.add_argument("--cnn-only", dest="use_morph", action="store_false")
    parser.add_argument("--target-condition", default="Light")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-cells-per-bag", type=int, default=500)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    root = os.path.dirname(os.path.dirname(__file__))
    path_cnn = os.path.join(root, "data", "cnn_features_pca.csv")
    path_morph = os.path.join(root, "data", "morph_features.csv")
    path_tgt = os.path.join(root, "data", "result.csv")

    # ---- hyperparams ----
    window_size = args.window_size
    predict_last = True
    n_cells_per_bag = args.n_cells_per_bag
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    mc_test_runs = 20
    mc_test_base_seed = 1000

    # model ablations
    model_type = args.model_type  # "odernn", "gru", "lstm"
    fusion_type = args.fusion_type  # "cross_attention" or "concat"
    use_morph = args.use_morph
    use_cnn = args.use_cnn

    # condition to train on
    target_condition = args.target_condition

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
    # early stopping & scheduler
    patience = 15
    bad_epochs = 0
    best_val_norm = float("inf")
    best_val_raw = float("inf")
    
    # Scheduler params
    use_scheduler = True
    scheduler_factor = 0.5
    scheduler_patience = 5
    min_lr = 1e-6
    
    # Gradient clipping
    grad_clip = 1.0

    print(f"[Dataset] train={len(ds_train)} | val={len(ds_val)} | test={len(ds_test)}")

    # ---- target scaler from TRAIN only ----
    y_mean, y_std = compute_target_scaler(ds_train)
    print(f"[Target scaler]\n y_mean: {y_mean.tolist()}\n y_std : {y_std.tolist()}")

    # 保存 scaler
    run_dir = build_run_dir(
        root,
        args.run_name,
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
    
    if model_type == "odernn":
        model = FusionODERNNModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            hidden_dim=128,
            ode_hidden=128,
            dropout=0.1,
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
        ).to(device)
    elif model_type == "gru":
        model = FusionGRUModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            rnn_hidden=128,
            rnn_layers=1,
            dropout=0.1,
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
        ).to(device)
    elif model_type == "lstm":
        model = FusionLSTMModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            rnn_hidden=128,
            rnn_layers=1,
            dropout=0.1,
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr
        )

    # ---- train loop ----
    ckpt_path = os.path.join(run_dir, f"best_ode_{target_condition}.pt")

    target_names = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]

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
            target_names = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]

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


if __name__ == "__main__":
    main()
