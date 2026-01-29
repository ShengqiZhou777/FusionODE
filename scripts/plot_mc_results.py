from __future__ import annotations
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Predicted vs True scatter plots from MC runs with Error Bars.")
    parser.add_argument("--csv", required=True, help="Path to mc_preds_*.csv")
    parser.add_argument("--out", default=None, help="Output image path (default: same as csv with .png)")
    args = parser.parse_args()

    df = _load_csv(args.csv)
    
    # Expected columns: run, target, true, pred
    required_cols = ["run", "target", "true", "pred"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    targets = df["target"].unique()
    n_targets = len(targets)
    
    # Setup plot: 1 row, N columns
    # Increase figure size slightly to accommodate error bars clearly
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 6))
    if n_targets == 1:
        axes = [axes]
    
    for i, t_name in enumerate(targets):
        ax = axes[i]
        sub = df[df["target"] == t_name].copy()
        
        # We need to aggregate across runs for each sample.
        # Since 'true' value is unique per sample (usually), we can group by it.
        # However, to be safer (in case of duplicate true values for different samples),
        # we construct a 'sample_id' based on the row order within each run.
        # Assuming the CSV is ordered: R0[S0..SN], R1[S0..SN]...
        
        # Check consistency
        runs = sub["run"].unique()
        n_per_run = sub[sub["run"] == runs[0]].shape[0]
        
        # Assign sample_id based on modulo or cumcount
        # sub is essentially N_runs blocks of N_samples rows.
        # Let's sort by run just in case, though usually CSV is sorted.
        sub = sub.sort_values("run")
        
        # Create a sample index relative to the run
        sub["sample_idx"] = sub.groupby("run").cumcount()
        
        # Now aggregate
        agg = sub.groupby("sample_idx").agg({
            "true": "first",    # True value should be constant
            "pred": ["mean", "std"]
        })
        
        y_true = agg["true"]["first"].to_numpy()
        y_pred_mean = agg["pred"]["mean"].to_numpy()
        y_pred_std = agg["pred"]["std"].to_numpy()
        
        # Handle case where only 1 run, std is nan
        if np.isnan(y_pred_std).all():
            y_pred_std = np.zeros_like(y_pred_mean)
        
        # Identify Condition from filename
        fname = os.path.basename(args.csv)
        cond = "Unknown"
        if "Light" in fname: cond = "Light"
        elif "Dark" in fname: cond = "Dark"
        
        title_str = f"{cond} - {t_name}"

        # Calculate R2 on the MEAN prediction
        r2 = r2_score(y_true, y_pred_mean)
        
        # Scatter with Error Bars
        # Error bars: yerr
        # We use alpha for bars to not clutter too much
        # fmt='o' creates the scatter points
        ax.errorbar(
            y_true, 
            y_pred_mean, 
            yerr=y_pred_std, 
            fmt='o', 
            markersize=4, 
            capsize=3,    # Small caps on error bars
            alpha=0.6, 
            ecolor='gray', # Error bar color
            label=f"N={len(y_true)}\nErr=StdDev"
        )
        
        # Perfect line
        low = min(y_true.min(), y_pred_mean.min())
        high = max(y_true.max(), y_pred_mean.max())
        margin = (high - low) * 0.1
        low -= margin
        high += margin
        
        ax.plot([low, high], [low, high], "r--", lw=2, label="Perfect")
        
        ax.set_title(f"{title_str}\nR2(Mean)={r2:.4f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted (Mean ± Std)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = args.out or os.path.splitext(args.csv)[0] + ".png"
    plt.savefig(out_path, dpi=200)
    print(f"[Saved plot] {out_path}")

if __name__ == "__main__":
    main()
