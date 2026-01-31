# src/data/io.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _time_to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip()
        if s.endswith("h"):
            s = s[:-1]
        return float(s)
    return float(x)

def _norm_condition(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def read_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 要求列: time, condition, 4 targets
    df["time"] = df["time"].apply(_time_to_float)
    df["condition"] = df["condition"].apply(_norm_condition)
    return df

def read_cnn_raw(path: str) -> pd.DataFrame:
    # Read without header
    # Cols 0,1,2: time, condition, file
    # Cols 3..514: features
    df = pd.read_csv(path, header=None)
    
    # Rename columns manually
    feat_cols = [f"cnn_feat_{i}" for i in range(df.shape[1] - 3)]
    cols = ["time", "condition", "file"] + feat_cols
    df.columns = cols
    
    df["time"] = df["time"].apply(_time_to_float)
    df["condition"] = df["condition"].apply(_norm_condition)
    return df

def read_cnn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["time"] = df["time"].apply(_time_to_float)
    df["condition"] = df["condition"].apply(_norm_condition)
    return df

def read_morph(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["time"] = df["time"].apply(_time_to_float)
    df["condition"] = df["condition"].apply(_norm_condition)
    return df
