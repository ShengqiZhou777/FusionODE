# src/data/preprocess.py
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import torch


def _stable_rng(seed: int, *parts: object) -> np.random.Generator:
    """
    Create a deterministic RNG from a base seed and arbitrary parts.
    This avoids Python's randomized hash across runs.
    """
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    for p in parts:
        h.update(str(p).encode("utf-8"))
    seed_int = int(h.hexdigest()[:16], 16) % (2**32)
    return np.random.default_rng(seed_int)


def _safe_morph_stats(mg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute mean/std/median/skew/kurtosis with numerical safeguards.
    mg: [N, D]
    returns: [5*D] concatenated stats
    """
    if mg.size == 0:
        raise ValueError("Empty morph array after filtering; cannot compute stats.")

    mg = np.asarray(mg, dtype=np.float32)
    mu = np.nanmean(mg, axis=0)
    md = np.nanmedian(mg, axis=0)

    centered = mg - mu
    m2 = np.nanmean(centered ** 2, axis=0)
    sd = np.sqrt(np.maximum(m2, eps))
    m3 = np.nanmean(centered ** 3, axis=0)
    m4 = np.nanmean(centered ** 4, axis=0)

    skew = m3 / (m2 ** 1.5 + eps)
    kurt = m4 / (m2 ** 2 + eps) - 3.0  # excess kurtosis (pandas default)

    stats = np.concatenate([mu, sd, md, skew, kurt], axis=0)
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    return stats


def _resolve_common_cell_ids(df_m: pd.DataFrame, df_c: pd.DataFrame) -> np.ndarray:
    morph_ids = df_m["cell_id"].unique()
    cnn_ids = df_c["cell_id"].unique()
    return np.intersect1d(morph_ids, cnn_ids, assume_unique=False)


def build_trajectories(
    df_cnn: pd.DataFrame,
    df_morph: pd.DataFrame,
    df_target: pd.DataFrame,
    n_cells_per_bag: int = 500,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_strategy: Literal["cell", "time"] = "cell",
    window_size: int | None = None,
    target_shuffle: Literal["none", "within_condition"] = "none",
    seed: int = 0,
    split_seed: int | None = None,
    bag_seed: int | None = None,
    sample_with_replacement: bool = True,
    morph_cols_to_use: list[str] | None = None,
) -> tuple[dict, dict, dict]:
    """
    params:
      n_cells_per_bag: 每个时间点最多采样的细胞数 (Bag Size).
      split_seed: 控制 split 的随机种子（不设则回退到 seed）
      bag_seed: 控制 bag 采样的随机种子（不设则回退到 seed）
      split_strategy: "cell"=同一时间点内按细胞划分, "time"=按时间点整体划分
      window_size: split_strategy="time" 时用于窗口级别切分的时间窗长度
      target_shuffle: "none"=不打乱, "within_condition"=在同一condition内打乱时间点的target
    returns: (traj_train, traj_val, traj_test)
    """
    def _allocate_window_counts(n_windows: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
        if n_windows <= 0:
            return 0, 0, 0

        counts = [int(n_windows * r) for r in ratios]
        if counts[0] == 0:
            counts[0] = 1
        if n_windows >= 2 and counts[1] == 0:
            counts[1] = 1
        if n_windows >= 3 and counts[2] == 0:
            counts[2] = 1

        total = sum(counts)
        while total > n_windows:
            for idx in [0, 1, 2]:
                if total <= n_windows:
                    break
                min_allowed = 0
                if idx == 0 and n_windows >= 1:
                    min_allowed = 1
                if idx == 1 and n_windows >= 2:
                    min_allowed = 1
                if idx == 2 and n_windows >= 3:
                    min_allowed = 1
                if counts[idx] > min_allowed:
                    counts[idx] -= 1
                    total -= 1
        while total < n_windows:
            counts[0] += 1
            total += 1

        return counts[0], counts[1], counts[2]

    def _time_split_with_windows(
        ordered_keys: list[tuple[str, float]],
        ratios: tuple[float, float, float],
        win_size: int,
    ) -> dict[tuple[str, float], list[str]]:
        n_total = len(ordered_keys)
        n_windows = n_total - win_size + 1
        if n_windows <= 0:
            return {}

        n_train_w, n_val_w, n_test_w = _allocate_window_counts(n_windows, ratios)
        split_ranges = {
            "train": (0, n_train_w - 1) if n_train_w > 0 else None,
            "val": (n_train_w, n_train_w + n_val_w - 1) if n_val_w > 0 else None,
            "test": (n_train_w + n_val_w, n_windows - 1) if n_test_w > 0 else None,
        }

        time_split: dict[tuple[str, float], list[str]] = {}
        for split_name, window_range in split_ranges.items():
            if window_range is None:
                continue
            w_start, w_end = window_range
            t_start = w_start
            t_end = min(w_end + win_size - 1, n_total - 1)
            for idx in range(t_start, t_end + 1):
                key = ordered_keys[idx]
                time_split.setdefault(key, [])
                if split_name not in time_split[key]:
                    time_split[key].append(split_name)

        return time_split
    # --- 找列 ---
    # --- 找列 ---
    cnn_feat_cols = [c for c in df_cnn.columns if c.startswith("cnn_pca_") or c.startswith("cnn_feat_") or c.startswith("cnn_raw_")]
    if len(cnn_feat_cols) == 0:
        raise ValueError("cnn_features_pca.csv 中找不到 cnn_pca_*, cnn_feat_*, 或 cnn_raw_* 列")

    non_feat = {"time", "condition", "cell_id"}
    morph_feat_cols = [c for c in df_morph.columns if c not in non_feat]
    if morph_cols_to_use is not None:
        morph_feat_cols = [c for c in morph_feat_cols if c in morph_cols_to_use]
    
    if len(morph_feat_cols) == 0:
        raise ValueError("morph_features.csv 中找不到形态学特征列 (或被过滤后为空)")

    target_cols = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
    for c in target_cols:
        if c not in df_target.columns:
            raise ValueError(f"result.csv 缺少 target 列: {c}")

    if split_seed is None:
        split_seed = seed
    if bag_seed is None:
        bag_seed = seed

    # --- groupby ---
    cnn_groups = {(cond, t): g for (cond, t), g in df_cnn.groupby(["condition", "time"], dropna=True)}
    morph_groups = {(cond, t): g for (cond, t), g in df_morph.groupby(["condition", "time"], dropna=True)}
    if df_target.duplicated(subset=["condition", "time"]).any():
        dup = df_target[df_target.duplicated(subset=["condition", "time"], keep=False)]
        raise ValueError(f"result.csv 存在重复 condition/time:\n{dup[['condition','time']].head()}")
    tgt_groups = {(r["condition"], float(r["time"])): r for _, r in df_target.iterrows()}

    keys = sorted(set(cnn_groups) & set(morph_groups) & set(tgt_groups))
    if len(keys) == 0:
        raise ValueError("三张表交集为空")

    traj_train: dict = {}
    traj_val: dict = {}
    traj_test: dict = {}

    for cond in sorted({k[0] for k in keys}):
        cond_keys = sorted([k for k in keys if k[0] == cond], key=lambda x: x[1])
        target_map = {k: k for k in cond_keys}
        if target_shuffle == "within_condition":
            rng_shuffle = _stable_rng(split_seed, cond, "target_shuffle")
            shuffled = list(cond_keys)
            rng_shuffle.shuffle(shuffled)
            target_map = dict(zip(cond_keys, shuffled))

        # Containers
        data_containers = {
            "train": {"times": [], "morph": [], "bags": [], "targets": [], "cell_ids": []},
            "val":   {"times": [], "morph": [], "bags": [], "targets": [], "cell_ids": []},
            "test":  {"times": [], "morph": [], "bags": [], "targets": [], "cell_ids": []}
        }

        if split_strategy == "time":
            time_split: dict[tuple[str, float], list[str]] = {}
            if window_size is not None and window_size > 1:
                time_split = _time_split_with_windows(cond_keys, split_ratios, window_size)

            if not time_split:
                n_total = len(cond_keys)
                n_train = int(n_total * split_ratios[0])
                n_val = int(n_total * split_ratios[1])
                train_times = set(cond_keys[:n_train])
                val_times = set(cond_keys[n_train : n_train + n_val])
                test_times = set(cond_keys[n_train + n_val :])

                for key in train_times:
                    time_split[key] = ["train"]
                for key in val_times:
                    time_split[key] = ["val"]
                for key in test_times:
                    time_split[key] = ["test"]
        else:
            time_split = {}

        for (c, t) in cond_keys:
            df_m = morph_groups[(c, t)]
            df_c = cnn_groups[(c, t)]

            common_cells = _resolve_common_cell_ids(df_m, df_c)
            if len(common_cells) == 0:
                continue

            if split_strategy == "time":
                split_names = time_split.get((c, t))
                if not split_names:
                    continue
                split_assignments = [(name, set(common_cells)) for name in split_names]
            else:
                rng_split = _stable_rng(split_seed, c, t, "split")
                rng_split.shuffle(common_cells)

                # Split cells
                n_total = len(common_cells)
                n_train = int(n_total * split_ratios[0])
                n_val = int(n_total * split_ratios[1])
                # n_test = rest

                train_cells = set(common_cells[:n_train])
                val_cells = set(common_cells[n_train : n_train + n_val])
                test_cells = set(common_cells[n_train + n_val :])
                split_assignments = [("train", train_cells), ("val", val_cells), ("test", test_cells)]

            for split_name, cell_set in split_assignments:
                if len(cell_set) == 0:
                    continue

                # Filter to current split
                df_m_sub = df_m[df_m["cell_id"].isin(cell_set)]
                df_c_sub = df_c[df_c["cell_id"].isin(cell_set)]

                # Subsample cell_ids if needed (shared between Morph and CNN)
                # Subsample cell_ids if needed (shared between Morph and CNN)
                # Optimization: Use set intersection instead of np.intersect1d to avoid repeated sorting
                ids_m = set(df_m_sub["cell_id"])
                ids_c = set(df_c_sub["cell_id"])
                available_ids = np.array(list(ids_m.intersection(ids_c)))
                n_current = len(available_ids)
                if n_current == 0:
                    continue

                rng_sample = _stable_rng(bag_seed, c, t, split_name)
                if n_current >= n_cells_per_bag:
                    kept_ids = rng_sample.choice(available_ids, size=n_cells_per_bag, replace=False)
                else:
                    if sample_with_replacement:
                        kept_ids = rng_sample.choice(available_ids, size=n_cells_per_bag, replace=True)
                    else:
                        kept_ids = available_ids
                
                # Apply selection to both (preserve duplicates if sampled with replacement)
                df_m_sub = df_m_sub.set_index("cell_id").loc[kept_ids].reset_index()
                df_c_sub = df_c_sub.set_index("cell_id").loc[kept_ids].reset_index()

                # Morph
                mg = df_m_sub[morph_feat_cols].astype(float).to_numpy()
                morph_vec = _safe_morph_stats(mg)

                # CNN bags
                cg = df_c_sub[cnn_feat_cols].astype(float).to_numpy()
                c_ids = df_c_sub["cell_id"].to_numpy()
                
                # Already subsampled via DataFrame iloc above
                bag_tensor = torch.tensor(cg, dtype=torch.float32)

                # Target
                tr = tgt_groups[target_map[(c, float(t))]]
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
