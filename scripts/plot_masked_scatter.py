import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing test_predictions_masked.csv")
    parser.add_argument("--output", type=str, default="masked_scatter.png", help="Output filename")
    args = parser.parse_args()

    csv_path = os.path.join(args.run_dir, "test_predictions_masked.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    targets = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
    
    # Setup plot 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    run_name = os.path.basename(args.run_dir)
    
    for i, t_name in enumerate(targets):
        ax = axes[i]
        
        col_true = f"True_{t_name}"
        col_pred = f"Pred_{t_name}"
        
        if col_true not in df.columns or col_pred not in df.columns:
            ax.text(0.5, 0.5, f"Missing cols for {t_name}", ha='center')
            continue
            
        y_true = df[col_true].values
        y_pred = df[col_pred].values
        
        # Metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='none', c='blue')
        
        # Perfect line
        low = min(y_true.min(), y_pred.min())
        high = max(y_true.max(), y_pred.max())
        margin = (high - low) * 0.1
        low -= margin
        high += margin
        ax.plot([low, high], [low, high], "r--", lw=2)
        
        ax.set_title(f"{t_name}\nR2={r2:.4f} | RMSE={rmse:.4f}")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted (Masked)")
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.suptitle(f"Masked Prediction Performance - {run_name}", fontsize=16)
    plt.tight_layout()
    
    out_path = os.path.join(args.run_dir, args.output)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
