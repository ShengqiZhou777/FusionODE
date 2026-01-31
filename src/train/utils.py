
import torch
from torch.utils.data import Subset

def compute_target_scaler(ds_subset: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of targets (4 dimensions) from a dataset subset.
    Returns: y_mean [4], y_std [4] (torch.float32)
    """
    ys = []
    # We iterate 
    for i in range(len(ds_subset)):
        item = ds_subset[i]
        ys.append(item["y"].float().unsqueeze(0))  # [1,4]
    Y = torch.cat(ys, dim=0)  # [N,4]
    y_mean = Y.mean(dim=0)
    y_std = Y.std(dim=0, unbiased=False)
    min_std = 0.05 * y_std.max()   # Avoid zero division or too small std
    y_std = y_std.clamp(min=min_std).clamp(min=1e-6)
    return y_mean, y_std
