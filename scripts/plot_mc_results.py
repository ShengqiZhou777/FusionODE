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
    parser = argparse.ArgumentParser(description="Plot Predicted vs True scatter plots from MC runs.")
    parser.add_argument("--csv", required=True, help="Path to mc_preds_*.csv")
    parser.add_argument("--out", default=None, help="Output image path (default: same as csv with .png)")
    args = parser.parse_args()

    df = _load_csv(args.csv)
    
    # Expected columns: run, target, true, pred
    required_cols = ["target", "true", "pred"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    targets = df["target"].unique()
    n_targets = len(targets)
    
    # Setup plot: 1 row, N columns
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]
    
    for i, t_name in enumerate(targets):
        ax = axes[i]
        sub = df[df["target"] == t_name]
        
        y_true = sub["true"].to_numpy()
        y_pred = sub["pred"].to_numpy()
        
        # Identify Condition from filename if possible, simply use t_name for title
        # "Light - Dry_Weight" style
        # Maybe parse condition from filename? mc_preds_Light_...
        fname = os.path.basename(args.csv)
        cond = "Unknown"
        if "Light" in fname: cond = "Light"
        elif "Dark" in fname: cond = "Dark"
        
        title_str = f"{cond} - {t_name}"

        # Calculate R2
        r2 = r2_score(y_true, y_pred)
        
        # Scatter
        ax.scatter(y_true, y_pred, alpha=0.3, s=20, label=f"N={len(y_true)}")
        
        # Perfect line
        low = min(y_true.min(), y_pred.min())
        high = max(y_true.max(), y_pred.max())
        margin = (high - low) * 0.1
        low -= margin
        high += margin
        
        ax.plot([low, high], [low, high], "r--", lw=2, label="Perfect")
        
        ax.set_title(f"{title_str}\nR2={r2:.4f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = args.out or os.path.splitext(args.csv)[0] + ".png"
    plt.savefig(out_path, dpi=200)
    print(f"[Saved plot] {out_path}")

if __name__ == "__main__":
    main()
