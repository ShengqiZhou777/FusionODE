# src/data/dataset.py
from __future__ import annotations
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, trajectories: dict, window_size: int = 5, predict_last: bool = True):
        self.traj = trajectories
        self.window_size = window_size
        self.predict_last = predict_last

        self.index = []
        W = window_size
        for cid, data in trajectories.items():
            T = len(data["times"])
            for s in range(0, T - W + 1):
                self.index.append((cid, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cid, s = self.index[idx]
        data = self.traj[cid]
        W = self.window_size

        times = data["times"][s:s+W]
        morph = data["morph"][s:s+W]
        bags  = data["bags"][s:s+W]
        yseq  = data["targets"][s:s+W]
        y = yseq[-1] if self.predict_last else yseq

        return {"cid": cid, "times": times, "morph": morph, "bags": bags, "y": y}
