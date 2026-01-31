
import os
import torch
from typing import Mapping, Any, Optional
from torch.utils.data import DataLoader, Subset

from src.data.io import read_cnn, read_morph, read_targets
from src.data.dataset import SlidingWindowDataset
from src.data.collate import collate_windows

def read_dataframes(root: str, data_cfg: Mapping[str, Any], use_raw_cnn: bool = False):
    """
    Read CNN, Morph, and Target CSVs.
    """
    if use_raw_cnn:
        default_cnn = os.path.join(root, "data", "cnn_features_resnet18.csv")
    else:
        default_cnn = os.path.join(root, "data", "cnn_features_pca.csv")
        
    path_cnn = data_cfg.get("cnn_path", default_cnn)
    path_morph = data_cfg.get("morph_path", os.path.join(root, "data", "morph_features.csv"))
    path_tgt = data_cfg.get("target_path", os.path.join(root, "data", "result.csv"))
    
    # If explicit path provided in data_cfg, use it. 
    # Logic in scripts/train.py was slightly different but this covers it.
    
    df_cnn = read_cnn(path_cnn)
    df_morph = read_morph(path_morph)
    df_tgt = read_targets(path_tgt)
    return df_cnn, df_morph, df_tgt

def _prepare_overfit_subset(ds_train: SlidingWindowDataset, ds_val: SlidingWindowDataset, overfit_n: int):
    """
    Optional overfit sanity check: train/val on the same tiny subset.
    """
    if overfit_n <= 0:
        return ds_train, ds_val
    subset_n = min(overfit_n, len(ds_train))
    subset_indices = list(range(subset_n))
    return Subset(ds_train, subset_indices), Subset(ds_train, subset_indices)

def create_datasets(
    traj_train: dict,
    traj_val: dict,
    traj_test: dict,
    window_size: int,
    predict_last: bool = True,
    overfit_n: int = 0,
) -> tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset]:
    ds_train = SlidingWindowDataset(traj_train, window_size=window_size, predict_last=predict_last)
    ds_val = SlidingWindowDataset(traj_val, window_size=window_size, predict_last=predict_last)
    ds_test = SlidingWindowDataset(traj_test, window_size=window_size, predict_last=predict_last)
    ds_train, ds_val = _prepare_overfit_subset(ds_train, ds_val, overfit_n)
    return ds_train, ds_val, ds_test

def create_loaders(
    ds_train: SlidingWindowDataset,
    ds_val: SlidingWindowDataset,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_windows,
        generator=generator,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_windows,
        generator=generator,
    )
    return dl_train, dl_val
