# src/data/preprocess.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch


def build_trajectories(
    df_cnn: pd.DataFrame,
    df_morph: pd.DataFrame,
    df_target: pd.DataFrame,
    n_cells_per_bag: int = 500,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> tuple[dict, dict, dict]:
    """
    params:
      n_cells_per_bag: 每个时间点最多采样的细胞数 (Bag Size).
    returns: (traj_train, traj_val, traj_test)
    """
    rng = np.random.default_rng(seed)

    # --- 找列 ---
    cnn_feat_cols = [c for c in df_cnn.columns if c.startswith("cnn_pca_")]
    if len(cnn_feat_cols) == 0:
        raise ValueError("cnn_features_pca.csv 中找不到 cnn_pca_* 列")

    non_feat = {"time", "condition", "cell_id"}
    morph_feat_cols = [c for c in df_morph.columns if c not in non_feat]
    if len(morph_feat_cols) == 0:
        raise ValueError("morph_features.csv 中找不到形态学特征列")

    target_cols = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
    for c in target_cols:
        if c not in df_target.columns:
            raise ValueError(f"result.csv 缺少 target 列: {c}")

    # --- groupby ---
    cnn_groups = {(cond, t): g for (cond, t), g in df_cnn.groupby(["condition", "time"], dropna=True)}
    morph_groups = {(cond, t): g for (cond, t), g in df_morph.groupby(["condition", "time"], dropna=True)}
    tgt_groups = {(r["condition"], float(r["time"])): r for _, r in df_target.iterrows()}

    keys = sorted(set(cnn_groups) & set(morph_groups) & set(tgt_groups))
    if len(keys) == 0:
        raise ValueError("三张表交集为空")

    traj_train: dict = {}
    traj_val: dict = {}
    traj_test: dict = {}

    for cond in sorted({k[0] for k in keys}):
        cond_keys = sorted([k for k in keys if k[0] == cond], key=lambda x: x[1])

        # Containers
        data_containers = {
            "train": {"times": [], "morph": [], "bags": [], "targets": [], "cell_ids": []},
            "val":   {"times": [], "morph": [], "bags": [], "targets": [], "cell_ids": []},
            "test":  {"times": [], "morph": [], "bags": [], "targets": [], "cell_ids": []}
        }

        for (c, t) in cond_keys:
            df_m = morph_groups[(c, t)]
            all_cells = df_m["cell_id"].unique()
            rng.shuffle(all_cells)

            # Split cells
            n_total = len(all_cells)
            n_train = int(n_total * split_ratios[0])
            n_val = int(n_total * split_ratios[1])
            # n_test = rest

            train_cells = set(all_cells[:n_train])
            val_cells = set(all_cells[n_train : n_train + n_val])
            test_cells = set(all_cells[n_train + n_val :])

            for split_name, cell_set in [("train", train_cells), ("val", val_cells), ("test", test_cells)]:
                if len(cell_set) == 0:
                    continue

                # Filter to current split
                df_m_sub = df_m[df_m["cell_id"].isin(cell_set)]
                # Note: df_c_sub filtering deferred to be synced with df_m_sub

                # Subsample cell_ids if needed (shared between Morph and CNN)
                available_ids = df_m_sub["cell_id"].to_numpy()
                n_current = len(available_ids)
                
                if n_current > n_cells_per_bag:
                    # Randomly select cell_ids
                    kept_ids = rng.choice(available_ids, size=n_cells_per_bag, replace=False)
                else:
                    kept_ids = available_ids
                
                # Apply selection to both
                df_m_sub = df_m_sub[df_m_sub["cell_id"].isin(kept_ids)]
                df_c_sub = cnn_groups[(c, t)]
                df_c_sub = df_c_sub[df_c_sub["cell_id"].isin(kept_ids)]

                # Morph
                mg = df_m_sub[morph_feat_cols].astype(float)
                mu = mg.mean(axis=0).to_numpy()
                sd = mg.std(axis=0, ddof=0).to_numpy()
                md = mg.median(axis=0).to_numpy()
                sk = mg.skew(axis=0).fillna(0.0).to_numpy()
                ku = mg.kurtosis(axis=0).fillna(0.0).to_numpy()
                morph_vec = np.concatenate([mu, sd, md, sk, ku], axis=0)

                # CNN bags
                cg = df_c_sub[cnn_feat_cols].astype(float).to_numpy()
                c_ids = df_c_sub["cell_id"].to_numpy()
                
                # Already subsampled via DataFrame iloc above
                bag_tensor = torch.tensor(cg, dtype=torch.float32)

                # Target
                tr = tgt_groups[(c, float(t))]
                y = np.array([tr[k] for k in target_cols], dtype=np.float32)

                dc = data_containers[split_name]
                dc["times"].append(float(t))
                dc["morph"].append(morph_vec)
                dc["bags"].append(bag_tensor)
                dc["targets"].append(y)
                dc["cell_ids"].append(c_ids)

        # Pack
        for split_name, out_dict in [("train", traj_train), ("val", traj_val), ("test", traj_test)]:
            dc = data_containers[split_name]
            if len(dc["times"]) > 0:
                out_dict[cond] = {
                    "times": torch.tensor(dc["times"], dtype=torch.float32),
                    "morph": torch.tensor(np.stack(dc["morph"]), dtype=torch.float32),
                    "bags": dc["bags"],
                    "targets": torch.tensor(np.stack(dc["targets"]), dtype=torch.float32),
                    "cell_ids": dc["cell_ids"],
                }

    return traj_train, traj_val, traj_test

