# src/data/preprocess.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

def filter_cell_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """你的标准细胞筛选规则（只依赖 morph 表字段）"""
    if "area" in df.columns:
        df = df[
            (df["area"] >= 500) &
            (df["area"] <= 3500) &
            (df["mean_intensity"] >= 80) &
            (df["mean_intensity"] <= 140) &
            (df["circularity"] >= 0.88)
        ]
    return df


def apply_cell_filter_to_both(df_morph: pd.DataFrame, df_cnn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    用 morph 的筛选结果，同步过滤 cnn，保证两模态细胞集合一致。
    优先使用 cell_id；如果没有，就尝试 (condition,time,file,cell_id) 等组合。
    """
    df_morph_f = filter_cell_dataframe(df_morph)

    # 使用唯一标识符同步过滤
    keys = ["condition", "time", "cell_id"]
    if not (all(col in df_morph_f.columns for col in keys) and all(col in df_cnn.columns for col in keys)):
        raise ValueError(
            "无法将 morph 的筛选同步到 cnn：两表缺少共同的字段 ['condition', 'time', 'cell_id']。"
        )

    keep = df_morph_f[keys].drop_duplicates()
    df_cnn_f = df_cnn.merge(keep, on=keys, how="inner")

    return df_morph_f, df_cnn_f


def build_trajectories(
    df_cnn: pd.DataFrame,
    df_morph: pd.DataFrame,
    df_target: pd.DataFrame,
    max_cells: int | None = 512,
    seed: int = 0,
) -> dict:
    """
    返回:
      trajectories[cond] = {
        "times": Tensor[T],
        "morph": Tensor[T, 5*Dm],
        "bags": list[T] of Tensor[Ni, Dc],
        "targets": Tensor[T, 4],
      }
    """
    df_morph, df_cnn = apply_cell_filter_to_both(df_morph, df_cnn)
    rng = np.random.default_rng(seed)

    # --- 找列 ---
    cnn_feat_cols = [c for c in df_cnn.columns if c.startswith("cnn_pca_")]
    if len(cnn_feat_cols) == 0:
        raise ValueError("cnn_features_pca.csv 中找不到 cnn_pca_* 列")

    # morph里排除非特征列（你如果还有别的meta列，加入这里排除）
    non_feat = {"time", "condition", "cell_id"}
    morph_feat_cols = [c for c in df_morph.columns if c not in non_feat]
    if len(morph_feat_cols) == 0:
        raise ValueError("morph_features.csv 中找不到形态学特征列（排除 time/condition/cell_id 后为空）")

    # target列名固定成这4个（如果你文件里名字不一样，就在这里改映射）
    target_cols = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
    for c in target_cols:
        if c not in df_target.columns:
            raise ValueError(f"result.csv 缺少 target 列: {c}")

    # --- groupby 构建字典 ---
    cnn_groups = {(cond, t): g for (cond, t), g in df_cnn.groupby(["condition", "time"], dropna=True)}
    morph_groups = {(cond, t): g for (cond, t), g in df_morph.groupby(["condition", "time"], dropna=True)}
    tgt_groups = {(r["condition"], float(r["time"])): r for _, r in df_target.iterrows()}

    keys = sorted(set(cnn_groups) & set(morph_groups) & set(tgt_groups))
    if len(keys) == 0:
        raise ValueError("三张表按 (condition,time) 对齐后交集为空：请检查 time 格式、condition 字符串")

    trajectories: dict = {}
    for cond in sorted({k[0] for k in keys}):
        cond_keys = sorted([k for k in keys if k[0] == cond], key=lambda x: x[1])

        times = []
        morph_stats = []
        bags = []
        targets = []

        for (c, t) in cond_keys:
            times.append(float(t))

            # ---- morph: 5 stats (mean, std, median, skew, kurt) ----
            mg = morph_groups[(c, t)][morph_feat_cols].astype(float)
            mu = mg.mean(axis=0).to_numpy()
            sd = mg.std(axis=0, ddof=0).to_numpy()
            md = mg.median(axis=0).to_numpy()
            sk = mg.skew(axis=0).fillna(0.0).to_numpy()
            ku = mg.kurtosis(axis=0).fillna(0.0).to_numpy()
            morph_stats.append(np.concatenate([mu, sd, md, sk, ku], axis=0))

            # ---- cnn bag + subsample ----
            cg = cnn_groups[(c, t)][cnn_feat_cols].astype(float).to_numpy()  # [N, Dc]
            if max_cells is not None and cg.shape[0] > max_cells:
                idx = rng.choice(cg.shape[0], size=max_cells, replace=False)
                cg = cg[idx]
            bags.append(torch.tensor(cg, dtype=torch.float32))

            # ---- targets ----
            tr = tgt_groups[(c, float(t))]
            y = np.array([tr[k] for k in target_cols], dtype=np.float32)
            targets.append(y)

        trajectories[cond] = {
            "times": torch.tensor(times, dtype=torch.float32),
            "morph": torch.tensor(np.stack(morph_stats), dtype=torch.float32),
            "bags": bags,
            "targets": torch.tensor(np.stack(targets), dtype=torch.float32),
        }

    return trajectories

