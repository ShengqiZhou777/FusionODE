# scripts/sanity_check.py
from __future__ import annotations
import os
import sys

# Ensure project root is in sys.path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import torch
from torch.utils.data import DataLoader

from src.data.io import read_cnn, read_morph, read_targets
from src.data.preprocess import build_trajectories
from src.data.dataset import SlidingWindowDataset
from src.data.collate import collate_windows

from src.models.full_model import FusionGRUModel
import torch.nn.functional as F

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    path_cnn = os.path.join(root, "data", "cnn_features_pca.csv")
    path_morph = os.path.join(root, "data", "morph_features.csv")
    path_tgt = os.path.join(root, "data", "result.csv")

    df_cnn = read_cnn(path_cnn)
    df_morph = read_morph(path_morph)
    df_tgt = read_targets(path_tgt)

    print("[CSV]")
    print(" cnn:", df_cnn.shape, "conds:", df_cnn["condition"].nunique(), "time[min,max]:", (df_cnn["time"].min(), df_cnn["time"].max()))
    print(" morph:", df_morph.shape, "conds:", df_morph["condition"].nunique(), "time[min,max]:", (df_morph["time"].min(), df_morph["time"].max()))
    print(" tgt:", df_tgt.shape, "conds:", df_tgt["condition"].nunique(), "time[min,max]:", (df_tgt["time"].min(), df_tgt["time"].max()))

    traj = build_trajectories(df_cnn, df_morph, df_tgt, max_cells=512, seed=0)

    print("\n[Trajectories]")
    for cond, d in traj.items():
        T = len(d["times"])
        Ns = [b.shape[0] for b in d["bags"]]
        print(f" {cond}: T={T}, morph_dim={d['morph'].shape[1]}, cnn_dim={d['bags'][0].shape[1]}, N(min/med/max)={min(Ns)}/{sorted(Ns)[len(Ns)//2]}/{max(Ns)}")

    ds = SlidingWindowDataset(traj, window_size=5, predict_last=True)
    print("\n[Dataset] num_samples:", len(ds))
    x0 = ds[0]
    print(" sample0 times:", x0["times"].tolist(), "morph:", tuple(x0["morph"].shape), "bag0:", tuple(x0["bags"][0].shape), "y:", tuple(x0["y"].shape))

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_windows, num_workers=0)
    batch = next(iter(dl))
    print("\n[Batch]")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f" {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f" {k}: {type(v)}")

    # Model Test
    print("\n[Model Test]")
    Dm = batch["morph"].shape[-1]
    Dc = batch["bags"].shape[-1]

    model = FusionGRUModel(morph_dim=Dm, cnn_dim=Dc)
    y_hat = model(batch)

    print(" y_hat:", y_hat.shape)   # 期望 [B,4]

    loss = F.mse_loss(y_hat, batch["y"])
    loss.backward()
    print(" loss:", float(loss))

if __name__ == "__main__":
    main()
