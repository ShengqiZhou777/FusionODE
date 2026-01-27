# src/data/collate.py
from __future__ import annotations
import torch

def collate_windows(batch):
    B = len(batch)
    W = batch[0]["times"].shape[0]
    Dc = batch[0]["bags"][0].shape[1]

    times = torch.stack([b["times"] for b in batch], dim=0)   # [B,W]
    morph = torch.stack([b["morph"] for b in batch], dim=0)   # [B,W,Dm]
    y = torch.stack([b["y"] for b in batch], dim=0)           # [B,4] or [B,W,4]

    Nmax = max(b["bags"][t].shape[0] for b in batch for t in range(W))

    bags_padded = torch.zeros(B, W, Nmax, Dc, dtype=torch.float32)
    bag_mask = torch.zeros(B, W, Nmax, dtype=torch.bool)

    for i, b in enumerate(batch):
        for t in range(W):
            bag = b["bags"][t]
            Ni = bag.shape[0]
            bags_padded[i, t, :Ni] = bag
            bag_mask[i, t, :Ni] = True

    return {"times": times, "morph": morph, "bags": bags_padded, "bag_mask": bag_mask, "y": y}
