# scripts/sanity_check.py
from __future__ import annotations
import os
import sys

# Ensure project root is in sys.path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.io import read_cnn, read_morph, read_targets
from src.data.preprocess import build_trajectories
from src.data.dataset import SlidingWindowDataset
from src.data.collate import collate_windows
from src.models.full_model import FusionGRUModel
from src.utils.seed import set_seed

def main():
    set_seed(0)

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

    # Updated to handle 3-way split return
    traj_train, traj_val, traj_test = build_trajectories(
        df_cnn, df_morph, df_tgt, n_cells_per_bag=500, seed=42
    )
    
    print("\n[Trajectory Stats & Samples]")
    all_splits = {"Train": traj_train, "Val": traj_val, "Test": traj_test}
    
    for split_name, traj_dict in all_splits.items():
        print(f"\n--- {split_name} Split ---")
        for cond, d in traj_dict.items():
            T = len(d["times"])
            Ns = [b.shape[0] for b in d["bags"]]
            print(f" {cond}: T={T}, N(min/med/max)={min(Ns)}/{sorted(Ns)[len(Ns)//2]}/{max(Ns)}")
            
            # Print sample cell IDs from the first timepoint
            if "cell_ids" in d and len(d["cell_ids"]) > 0:
                t0_ids = d["cell_ids"][0]  # Array of IDs for t=0
                sample = t0_ids[:10]
                print(f"   Sample IDs (t=0): {sample}") 
                print(f"   Total cells at t=0: {len(t0_ids)}")

    # Use train set for dataset check
    ds = SlidingWindowDataset(traj_train, window_size=5, predict_last=True)
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
    
    # --- Inspect CNN Aggregation ---
    print("\n[CNN Encoder Check]")
    z_c = model.cnn_encoder(batch["bags"], batch["bag_mask"])
    print(f" Input Bags: {batch['bags'].shape}")
    print(f" Output z_c: {z_c.shape}")  # Should be [B, W, z_cnn]
    print(f" Sample0, t=0 Aggregated Vector (first 12 dims):\n {z_c[0, 0, :12].detach().cpu().numpy()}")
    # -------------------------------

    y_hat = model(batch)

    print(" y_hat:", y_hat.shape)   # 期望 [B,4]

    loss = F.mse_loss(y_hat, batch["y"])
    loss.backward()
    print(" loss:", float(loss))

    # --- Compare Splits Aggregation ---
    print("\n[Compare Splits: Dark @ t=0]")
    cond = "Dark"
    t_idx = 0
    splits = {"Train": traj_train, "Val": traj_val, "Test": traj_test}
    
    for name, traj in splits.items():
        if cond in traj:
            # Get single bag: [500, 64]
            bag = traj[cond]["bags"][t_idx] 
            # Make batch: [1, 1, 500, 64]
            input_bag = bag.unsqueeze(0).unsqueeze(0).to(batch["bags"].device) # ensure same device if needed, but here likely CPU
            mask = torch.ones((1, 1, 500), dtype=torch.bool)
            
            # Encoder
            # Note: Model weights are random here, so embedding is random projection, 
            # but we can see if different cell populations yield similar or distinct vectors.
            z = model.cnn_encoder(input_bag, mask) # [1, 1, 64]
            vec = z[0, 0, :8].detach().numpy()
            print(f" {name:5s} Agg (first 8 dims): {vec}")

if __name__ == "__main__":
    main()
