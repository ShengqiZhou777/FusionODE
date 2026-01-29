#!/usr/bin/env python
"""
Summarize ablation results and generate publication-quality plots.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and visualize ablation results.")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root directory containing run subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save plots and tables.",
    )
    return parser.parse_args()


def parse_run_dir(dirname: str) -> dict | None:
    """
    Parses run directory name to extract config.
    Handles standard ablation and masked interpolation formats.
    """
    parts = dirname.split("_")
    
    # Needs at least timestamp (2 parts: date_time) + some config
    if len(parts) < 4:
        return None

    # Determine Masking Strategy
    is_masked = "masked_interp" in dirname
    strategy = "Masked Interpolation" if is_masked else "Standard Training"

    # Extract Window (always at end, wX)
    window_part = parts[-1]
    if not window_part.startswith("w"):
        return None
    try:
        window = int(window_part[1:])
    except ValueError:
        return None

    # Condition (always before Window)
    condition = parts[-2]
    
    # Middle part for config parsing
    # Standard: date_time_model_fusion_modality_cond_win
    # Masked: date_time_masked_interp_model_fusion_cond_win (modality implicit)
    
    middle_str = dirname.lower()

    # Determine Backbone
    if "odernn" in middle_str:
        model = "odernn"
    elif "lstm" in middle_str:
        model = "lstm"
    elif "gru" in middle_str:
        model = "gru"
    else:
        model = "unknown"
        
    # Determine Fusion Type
    if "cross_attention" in middle_str:
        fusion = "cross_attention"
    elif "concat" in middle_str:
        fusion = "concat"
    else:
        fusion = "unknown"
        
    # Determine Modality
    # Masked runs don't have modality in name, they imply Fusion
    if is_masked:
        modality = "morph+cnn"
    else:
        if "morph+cnn" in middle_str:
            modality = "morph+cnn"
        elif "morph" in middle_str:
            modality = "morph"
        elif "cnn" in middle_str:
            modality = "cnn"
        else:
            modality = "unknown"

    return {
        "dir": dirname,
        "Model": model,
        "Fusion": fusion,
        "Modality": modality,
        "Condition": condition,
        "Window": window,
        "Strategy": strategy,
        "raw_dir": dirname
    }


def load_data(runs_dir: Path) -> pd.DataFrame:
    data = []
    if not runs_dir.exists():
        print(f"[Error] Runs directory not found: {runs_dir}")
        return pd.DataFrame()

    subdirs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])
    print(f"[Info] Found {len(subdirs)} directories in {runs_dir}")

    for d in subdirs:
        info = parse_run_dir(d)
        if not info:
             continue

        # Look for metrics file
        # Standard: mc_metrics_*.csv
        # Masked: We didn't save mc_metrics in masked script, only printed.
        # Wait, the UPDATED masked script only saves test_predictions_masked.csv and scaler.pt
        # It does NOT save a metrics summary CSV yet.
        # However, checking the user output from Step 1966, it printed [Test Result].
        # But for summary, we need a FILE. 
        # If no file exists, we skip it for now (or I need to update masked script to save metrics too).
        # Let's check for mc_metrics first.
        
        csv_files = glob.glob(str(runs_dir / d / "mc_metrics_*.csv"))
        if not csv_files:
            # Maybe it's a masked run? Check for something else or skip
            # For now, let's look for any .csv that looks like metrics?
            # Actually, standard runs save mc_metrics. Masked runs currently don't save a summary metrics file.
            # I must rely on what's available. 
            # If I want to include Masked runs in the table, I need to compute metrics from predictions csv
            pred_csv = runs_dir / d / "test_predictions_masked.csv"
            if info["Strategy"] == "Masked Interpolation" and pred_csv.exists():
                # Compute on the fly
                from sklearn.metrics import r2_score, mean_squared_error
                import numpy as np
                df_pred = pd.read_csv(pred_csv)
                targets = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
                row_metrics = {}
                for tgt in targets:
                    y_true = df_pred[f"True_{tgt}"]
                    y_pred = df_pred[f"Pred_{tgt}"]
                    row_metrics[f"r2_{tgt}"] = r2_score(y_true, y_pred)
                    row_metrics[f"rmse_raw_{tgt}"] = np.sqrt(mean_squared_error(y_true, y_pred))
                
                # Treat as single run (mean=value, std=0)
                # Or if we have multiple masked runs, we can aggregate later.
                # Here we format structure to match
                for tgt in targets:
                    for met in ["r2", "rmse_raw"]:
                        col = f"{met}_{tgt}"
                        data.append({
                            **info,
                            "Target": tgt,
                            "Metric": met,
                            "Mean": row_metrics.get(col, 0.0),
                            "Std": 0.0, # Single run
                            "RunDir": d
                        })
                continue
            else:
                # print(f"Skipped {d}: no metrics CSV found")
                continue
            
        # Standard flow
        df_metrics = pd.read_csv(csv_files[0])
        means = df_metrics.mean(numeric_only=True)
        stds = df_metrics.std(numeric_only=True)

        targets = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
        metrics = ["r2", "rmse_raw", "mae_raw"]

        for tgt in targets:
            for met in metrics:
                col = f"{met}_{tgt}"
                if col in means:
                    data.append({
                        **info,
                        "Target": tgt,
                        "Metric": met,
                        "Mean": means[col],
                        "Std": stds[col],
                        "RunDir": d
                    })

    return pd.DataFrame(data)


def create_markdown_table(df_subset, index_cols, col_name, val_col="StrVal"):
    """
    Generic function to create a markdown table pivoted from dataframe.
    """
    if df_subset.empty:
        return "No Data"
        
    # Aggregate (mean of means if duplicates)
    agg = df_subset.groupby(index_cols + [col_name], as_index=False).agg({
        "Mean": "mean", "Std": "mean"
    })
    
    agg["StrVal"] = agg.apply(lambda r: f"{r['Mean']:.3f} ± {r['Std']:.3f}", axis=1)
    
    pivot = agg.pivot(index=index_cols, columns=col_name, values="StrVal")
    df_pivot = pivot.reset_index()
    
    columns = df_pivot.columns.tolist()
    header = "| " + " | ".join([str(c) for c in columns]) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    rows = []
    for _, r in df_pivot.iterrows():
        # Handle MultiIndex logic if needed, but here simple
        row_str = "| " + " | ".join([str(r[c]).replace("nan", "-") for c in columns]) + " |"
        rows.append(row_str)
        
    return "\n".join([header, sep] + rows)


def generate_full_report(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ablation_summary_detailed.md"
    
    lines = ["# Comprehensive Ablation Study Results\n"]
    
    # Filter for R2 only for simplicity in main tables (can change to RMSE)
    df_r2 = df[df["Metric"] == "r2"]
    
    # Iterate Conditions conditions (Light/Dark)
    conditions = sorted(df_r2["Condition"].unique())
    windows = sorted(df_r2["Window"].unique())
    
    for cond in conditions:
        lines.append(f"\n## Condition: {cond}")
        
        for win in windows:
            subset = df_r2[(df_r2["Condition"] == cond) & (df_r2["Window"] == win)]
            if subset.empty:
                continue
            
            lines.append(f"\n### Window: {win}h (Condition: {cond})\n")
            
            # 1. Backbone Comparison
            # Fix: Fusion=CrossAttn, Modality=Morph+CNN, Strategy=Standard
            lines.append("#### 1. Temporal Backbone Comparison (Fusion=CrossAttn, Modality=Morph+CNN)")
            df_back = subset[
                (subset["Fusion"] == "cross_attention") & 
                (subset["Modality"] == "morph+cnn") &
                (subset["Strategy"] == "Standard Training")
            ]
            if not df_back.empty:
                lines.append(create_markdown_table(df_back, ["Model"], "Target"))
            else:
                lines.append("*No data available*")
            lines.append("\n")

            # 2. Component Ablation
            # Fix: Model=ODERNN, Fusion=CrossAttn (for fusion row), Strategy=Standard
            # We want rows: MorphOnly, CNNOnly, Fusion
            lines.append("#### 2. Component Ablation (Model=ODERNN)")
            # Filter for ODERNN Standard
            df_comp = subset[
                (subset["Model"] == "odernn") & 
                (subset["Strategy"] == "Standard Training")
            ]
            # Further filter: 
            # Row 1: Modality=Morph
            # Row 2: Modality=CNN
            # Row 3: Modality=Morph+CNN (Fusion=CrossAttn)
            # Row 4: Modality=Morph+CNN (Fusion=Concat) -- Optional, maybe separate table
            
            # Let's show all available modalities/fusions for ODERNN
            if not df_comp.empty:
                # Create a "Variant" column for the table
                def get_variant(row):
                    if row["Modality"] == "morph": return "Morph Only"
                    if row["Modality"] == "cnn": return "CNN Only"
                    if row["Modality"] == "morph+cnn": return f"Fusion ({row['Fusion']})"
                    return "Unknown"
                
                df_comp = df_comp.copy()
                df_comp["Variant"] = df_comp.apply(get_variant, axis=1)
                lines.append(create_markdown_table(df_comp, ["Variant"], "Target"))
            else:
                lines.append("*No data available*")
            lines.append("\n")

            # 3. Masking Strategy
            # Fix: Model=ODERNN, Fusion=CrossAttn, Modality=Morph+CNN
            lines.append("#### 3. Masking Strategy (ODERNN + CrossAttn)")
            df_mask = subset[
                (subset["Model"] == "odernn") &
                (subset["Fusion"] == "cross_attention") &
                (subset["Modality"] == "morph+cnn")
            ]
            if not df_mask.empty:
                 lines.append(create_markdown_table(df_mask, ["Strategy"], "Target"))
            else:
                lines.append("*No data available*")
            # 4. GRU Component Ablation (New)
            lines.append("#### 4. Component Ablation (Model=GRU)")
            df_gru = subset[
                (subset["Model"] == "gru") &
                (subset["Strategy"] == "Standard Training")
            ]
            if not df_gru.empty:
                # Create a "Variant" column for the table
                def get_variant_gru(row):
                    if row["Modality"] == "morph": return "Morph Only"
                    if row["Modality"] == "cnn": return "CNN Only"
                    if row["Modality"] == "morph+cnn": return f"Fusion ({row['Fusion']})"
                    return "Unknown"
                
                df_gru = df_gru.copy()
                df_gru["Variant"] = df_gru.apply(get_variant_gru, axis=1)
                lines.append(create_markdown_table(df_gru, ["Variant"], "Target"))
            else:
                lines.append("*No data available*")
            lines.append("\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
        
    print(f"Saved detailed report to {report_path}")
    print("\n" + "\n".join(lines)) # Print to console


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    runs_dir = root / args.runs_dir
    output_dir = root / args.output_dir
    
    print("Loading data...")
    df = load_data(runs_dir)
    
    if df.empty:
        print("No data found.")
        return
        
    print("Generating report...")
    generate_full_report(df, output_dir)
    
if __name__ == "__main__":
    main()
