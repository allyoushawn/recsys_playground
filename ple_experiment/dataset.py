from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


Split = Literal["train", "val", "test"]


class CensusKDDDataset(Dataset):
    """
    Loads features and labels from .npy files produced by prepare_census_income.py

    Returns a dict per item with keys:
      - "x": torch.float32 tensor of features
      - "y_income": torch.long tensor (0/1)
      - "y_never_married": torch.long tensor (0/1)
    """

    def __init__(self, data_dir: str, split: Split):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train','val','test'}")

        self.split = split
        self.X = np.load(f"{data_dir}/X_{split}.npy")
        self.y_income = np.load(f"{data_dir}/y_income_{split}.npy")
        self.y_never = np.load(f"{data_dir}/y_nevermarried_{split}.npy")

        if self.X.dtype != np.float32:
            raise TypeError("X must be float32")
        if self.y_income.dtype != np.uint8 or self.y_never.dtype != np.uint8:
            raise TypeError("y arrays must be uint8")
        if np.isnan(self.X).any():
            raise ValueError("Found NaNs in X")
        if not set(np.unique(self.y_income)).issubset({0, 1}):
            raise ValueError("y_income must be binary {0,1}")
        if not set(np.unique(self.y_never)).issubset({0, 1}):
            raise ValueError("y_nevermarried must be binary {0,1}")
        if not (len(self.X) == len(self.y_income) == len(self.y_never)):
            raise ValueError("X and y lengths do not match")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.X[idx]).to(torch.float32)
        yi = torch.tensor(int(self.y_income[idx]), dtype=torch.long)
        yn = torch.tensor(int(self.y_never[idx]), dtype=torch.long)
        return {"x": x, "y_income": yi, "y_never_married": yn}


def make_dataloaders(
    data_dir: str,
    batch_size: int = 4096,
    num_workers: int = 4,
) -> Dict[Split, DataLoader]:
    """Create train/val/test DataLoaders from saved .npy artifacts."""
    loaders: Dict[Split, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = CensusKDDDataset(data_dir=data_dir, split=split)  # type: ignore[arg-type]
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders

