from __future__ import annotations
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _summary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key in keys:
        if key not in df.columns:
            continue
        vals = df[key].to_numpy()
        rows.append({"metric": key, "mean": float(vals.mean()), "std": float(vals.std(ddof=0))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MC test metrics with mean±std error bars.")
    parser.add_argument("--csv", required=True, help="Path to mc_test_*.csv")
    parser.add_argument("--out", default=None, help="Output image path (default: same as csv with .png)")
    parser.add_argument("--metrics", default="r2_mean,rmse_raw_t0,rmse_raw_t1,rmse_raw_t2,rmse_raw_t3",
                        help="Comma-separated metric columns to plot.")
    args = parser.parse_args()

    df = _load_csv(args.csv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    summary = _summary(df, metrics)
    if summary.empty:
        raise ValueError("No matching metric columns found to plot.")

    out_path = args.out or os.path.splitext(args.csv)[0] + ".png"

    plt.figure(figsize=(10, 5))
    xs = range(len(summary))
    plt.bar(xs, summary["mean"], yerr=summary["std"], capsize=4, color="#4C78A8")
    plt.xticks(xs, summary["metric"], rotation=30, ha="right")
    plt.ylabel("Metric value")
    plt.title("MC Test Metrics (mean ± std)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[Saved plot] {out_path}")


if __name__ == "__main__":
    main()
