import sys
import os

# Ensure project root is in sys.path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from src.data.io import read_cnn, read_morph
from src.data.preprocess import apply_cell_filter_to_both
import pandas as pd

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # FusionODE/
    path_cnn = os.path.join(root, "data", "cnn_features_pca.csv")
    path_morph = os.path.join(root, "data", "morph_features.csv")

    df_cnn = read_cnn(path_cnn)
    df_morph = read_morph(path_morph)

    print(f"Original Morph: {len(df_morph)}")
    
    # Apply filter
    df_morph_f, _ = apply_cell_filter_to_both(df_morph, df_cnn)
    
    print(f"Filtered Morph: {len(df_morph_f)}")
    print("-" * 40)
    print(f"{'Condition':<10} | {'Time (h)':<8} | {'Cell Count':<10}")
    print("-" * 40)
    
    # Group by condition and time to count cells
    counts = df_morph_f.groupby(["condition", "time"]).size().reset_index(name="count")
    
    for _, row in counts.iterrows():
        print(f"{row['condition']:<10} | {row['time']:<8.1f} | {row['count']:<10}")

if __name__ == "__main__":
    main()
