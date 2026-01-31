import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import os
import argparse

def plot_scatter_comparison(csv_paths, labels, output_path="comparison_scatter.png"):
    """
    Plots scatter plots for multiple models comparing True vs Pred values.
    
    Args:
        csv_paths: List of paths to the prediction CSV files.
        labels: List of labels for each model (e.g., ["Fusion", "Morph", "CNN"]).
        output_path: Path to save the resulting plot.
    """
    # Define targets
    targets = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
    n_targets = len(targets)
    n_models = len(csv_paths)
    
    # Setup plot
    fig, axes = plt.subplots(n_targets, n_models, figsize=(5 * n_models, 4 * n_targets), squeeze=False)
    
    # Global range for each target to keep axes consistent
    ranges = {}

    # Load data first to determine ranges
    all_data = []
    for path in csv_paths:
        if os.path.exists(path):
            all_data.append(pd.read_csv(path))
        else:
            print(f"Warning: File not found: {path}")
            all_data.append(None)

    for i, target in enumerate(targets):
        true_col = f"True_{target}"
        pred_col = f"Pred_{target}"
        
        vals = []
        for df in all_data:
            if df is not None:
                vals.extend(df[true_col].values)
                vals.extend(df[pred_col].values)
        
        if vals:
            min_val, max_val = min(vals), max(vals)
            padding = (max_val - min_val) * 0.05
            ranges[target] = (min_val - padding, max_val + padding)
        else:
            ranges[target] = (0, 1)

    # Plotting
    for j, (df, label) in enumerate(zip(all_data, labels)):
        if df is None:
            continue
            
        for i, target in enumerate(targets):
            ax = axes[i, j]
            true_col = f"True_{target}"
            pred_col = f"Pred_{target}"
            
            y_true = df[true_col]
            y_pred = df[pred_col]
            
            # Metrics
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            
            # Scatter
            sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.5, edgecolor=None)
            
            # Diagonal line
            min_lim, max_lim = ranges[target]
            ax.plot([min_lim, max_lim], [min_lim, max_lim], 'r--', alpha=0.7)
            
            ax.set_xlim(min_lim, max_lim)
            ax.set_ylim(min_lim, max_lim)
            
            if i == 0:
                ax.set_title(f"{label}\n{target}", fontweight='bold')
            else:
                ax.set_title(target, fontweight='bold')
                
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")
            
            # Add metrics text
            ax.text(0.05, 0.95, f"$R^2={r2:.3f}$\nMSE={mse:.3f}", 
                    transform=ax.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison scatter plots.")
    parser.add_argument("--files", nargs='+', required=True, help="List of CSV files")
    parser.add_argument("--labels", nargs='+', required=True, help="List of labels for the files")
    parser.add_argument("--output", default="comparison_scatter.png", help="Output filename")
    
    args = parser.parse_args()
    
    if len(args.files) != len(args.labels):
        print("Error: Number of files must match number of labels.")
    else:
        plot_scatter_comparison(args.files, args.labels, args.output)
