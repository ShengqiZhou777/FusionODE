from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.io import read_cnn, read_morph, read_targets
from src.data.preprocess import build_trajectories
from src.data.dataset import SlidingWindowDataset
from src.data.collate import collate_windows
from src.models.full_model import FusionGRUModel, FusionLSTMModel, FusionODERNNModel
from src.train.losses import mse_loss
from src.train.metrics import batch_mse, summarize_regression_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with masked timepoint interpolation.")
    parser.add_argument("--model-type", choices=["odernn", "gru", "lstm"], default="odernn")
    parser.add_argument("--fusion-type", choices=["cross_attention", "concat"], default="cross_attention")
    parser.add_argument("--morph-only", dest="use_cnn", action="store_false", default=True)
    parser.add_argument("--cnn-only", dest="use_morph", action="store_false", default=True)
    parser.add_argument("--target-condition", default="Light")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-cells-per-bag", type=int, default=500)
    parser.add_argument("--rnn-hidden", type=int, default=128)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ode-hidden", type=int, default=128)
    parser.add_argument("--attn-dim", type=int, default=64)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--mask-prob-train", type=float, default=0.5)
    parser.add_argument("--mask-prob-eval", type=float, default=0.0)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_run_dir(root: str, run_name: str | None, tag: str) -> str:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{tag}"
    return os.path.join(root, "runs", run_name)


def compute_target_scaler(ds_subset: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    ys = []
    for i in range(len(ds_subset)):
        item = ds_subset[i]
        ys.append(item["y"].float().unsqueeze(0))
    Y = torch.cat(ys, dim=0)
    y_mean = Y.mean(dim=0)
    y_std = Y.std(dim=0, unbiased=False)
    min_std = 0.05 * y_std.max()
    y_std = y_std.clamp(min=min_std).clamp(min=1e-6)
    return y_mean, y_std


def apply_time_mask(batch: dict, rng: torch.Generator, prob: float) -> dict:
    if prob <= 0:
        return batch
    if torch.rand(1, generator=rng).item() > prob:
        return batch
    W = batch["times"].shape[1]
    if W <= 1:
        return batch
    t = int(torch.randint(0, W, (1,), generator=rng).item())
    batch["morph"][:, t] = 0.0
    batch["bags"][:, t] = 0.0
    batch["bag_mask"][:, t] = False
    return batch


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def eval_one_epoch_masked(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    rng: torch.Generator,
    mask_prob: float,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    target_names: list[str],
) -> dict:
    model.eval()
    all_y = []
    all_yhat = []
    mse_raw_list, mse_norm_list = [], []

    for batch in dataloader:
        batch = apply_time_mask(batch, rng, mask_prob)
        batch = to_device(batch, device)
        y_hat = model(batch)
        y = batch["y"]
        all_y.append(y.detach().cpu())
        all_yhat.append(y_hat.detach().cpu())
        mse_raw, mse_norm = batch_mse(y_hat, y, y_mean=y_mean, y_std=y_std)
        mse_raw_list.append(mse_raw)
        mse_norm_list.append(mse_norm)

    Y = torch.cat(all_y, dim=0)
    YH = torch.cat(all_yhat, dim=0)
    return summarize_regression_metrics(
        Y,
        YH,
        mse_raw_list,
        mse_norm_list,
        use_norm=True,
        target_names=target_names,
    )


def train_one_epoch_masked(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: torch.Generator,
    mask_prob: float,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    mse_raw_list, mse_norm_list = [], []
    for batch in dataloader:
        batch = apply_time_mask(batch, rng, mask_prob)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(batch)
        y = batch["y"]
        loss, mse_raw, mse_norm = mse_loss(y_hat, y, y_mean=y_mean, y_std=y_std)
        mse_raw_list.append(mse_raw.detach().item())
        mse_norm_list.append(mse_norm.detach().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    mse_raw = float(sum(mse_raw_list) / max(len(mse_raw_list), 1))
    mse_norm = float(sum(mse_norm_list) / max(len(mse_norm_list), 1))
    return {"mse_raw": mse_raw, "mse_norm": mse_norm}


def main() -> None:
    args = parse_args()
    set_seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    root = os.path.dirname(os.path.dirname(__file__))
    path_cnn = os.path.join(root, "data", "cnn_features_pca.csv")
    path_morph = os.path.join(root, "data", "morph_features.csv")
    path_tgt = os.path.join(root, "data", "result.csv")

    window_size = args.window_size
    predict_last = True
    n_cells_per_bag = args.n_cells_per_bag
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs

    model_type = args.model_type
    fusion_type = args.fusion_type
    use_morph = args.use_morph
    use_cnn = args.use_cnn
    target_condition = args.target_condition

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print(f"[Target Condition] {target_condition}")

    df_cnn = read_cnn(path_cnn)
    df_morph = read_morph(path_morph)
    df_tgt = read_targets(path_tgt)

    traj_train, traj_val, traj_test = build_trajectories(
        df_cnn,
        df_morph,
        df_tgt,
        n_cells_per_bag=n_cells_per_bag,
        split_ratios=(0.7, 0.15, 0.15),
        seed=0,
        split_seed=0,
        bag_seed=0,
        sample_with_replacement=True,
    )

    traj_train = {k: v for k, v in traj_train.items() if k == target_condition}
    traj_val = {k: v for k, v in traj_val.items() if k == target_condition}
    traj_test = {k: v for k, v in traj_test.items() if k == target_condition}

    ds_train = SlidingWindowDataset(traj_train, window_size=window_size, predict_last=predict_last)
    ds_val = SlidingWindowDataset(traj_val, window_size=window_size, predict_last=predict_last)
    ds_test = SlidingWindowDataset(traj_test, window_size=window_size, predict_last=predict_last)

    y_mean, y_std = compute_target_scaler(ds_train)

    run_dir = build_run_dir(
        root,
        args.run_name,
        f"masked_interp_{model_type}_{fusion_type}_{target_condition}_w{window_size}",
    )
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"y_mean": y_mean, "y_std": y_std}, os.path.join(run_dir, "scaler.pt"))

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

    batch0 = next(iter(dl_train))
    Dm = batch0["morph"].shape[-1]
    Dc = batch0["bags"].shape[-1]

    if model_type == "odernn":
        model = FusionODERNNModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            hidden_dim=args.hidden_dim,
            ode_hidden=args.ode_hidden,
            dropout=0.1,
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
            attn_dim=args.attn_dim,
            attn_heads=args.attn_heads,
        ).to(device)
    elif model_type == "gru":
        model = FusionGRUModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            rnn_hidden=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            dropout=0.1,
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
            attn_dim=args.attn_dim,
            attn_heads=args.attn_heads,
        ).to(device)
    elif model_type == "lstm":
        model = FusionLSTMModel(
            morph_dim=Dm,
            cnn_dim=Dc,
            z_morph=64,
            z_cnn=64,
            rnn_hidden=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            dropout=0.1,
            use_morph=use_morph,
            use_cnn=use_cnn,
            fusion_type=fusion_type,
            attn_dim=args.attn_dim,
            attn_heads=args.attn_heads,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    target_names = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]

    for ep in range(epochs):
        tr = train_one_epoch_masked(
            model,
            dl_train,
            optimizer,
            device,
            g,
            args.mask_prob_train,
            y_mean,
            y_std,
        )
        va = eval_one_epoch_masked(
            model,
            dl_val,
            device,
            g,
            args.mask_prob_eval,
            y_mean,
            y_std,
            target_names,
        )
        print(
            f"Epoch {ep:03d} | Train Loss={tr['mse_norm']:.4f} | "
            f"Val Loss={va['mse_norm']:.4f} Val MSE={va['mse_raw']:.4f}"
        )

    if len(ds_test) > 0:
        dl_test = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_windows,
            generator=g,
        )
        test_metrics = eval_one_epoch_masked(
            model,
            dl_test,
            device,
            g,
            args.mask_prob_eval,
            y_mean,
            y_std,
            target_names,
        )
        print(
            f"[Test Result] mse_raw={test_metrics['mse_raw']:.6f} | "
            f"mse_norm={test_metrics['mse_norm']:.6f}"
        )


if __name__ == "__main__":
    main()
