"""
Data loading for the offset-retrieval dataset.

The target at position i is the token at position i - k. Positions
i < k have no target and carry IGNORE_INDEX.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class OffsetDataset(Dataset):
    """Random token sequences, each target the token k positions earlier."""

    def __init__(self, data_dir: str, split: str, k: int):
        """
        Args:
            data_dir: Directory containing train.npz and test.npz
            split: 'train' or 'test'
            k: Offset between a position and the token it retrieves
        """
        data = np.load(os.path.join(data_dir, f'{split}.npz'))
        self.tokens = torch.from_numpy(data['tokens']).long()
        self.k = k
        self.targets = torch.full_like(self.tokens, IGNORE_INDEX)
        self.targets[:, k:] = self.tokens[:, :-k]

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tokens: Tensor [T] (long), 0-15
            targets: Tensor [T] (long), tokens[i - k] at i >= k and
                IGNORE_INDEX at i < k
        """
        return self.tokens[idx], self.targets[idx]
