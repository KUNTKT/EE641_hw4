"""
Data loading for the addition dataset.

Token ids: digits 0-9, PLUS = 10, BOS = 11. Operands are three
zero-padded digits and sums four, written most significant digit first.
The 'reversed' digit order reverses the sum only, never the operands.
"""

import json
import os

import torch
from torch.utils.data import Dataset

PLUS = 10
BOS = 11
VOCAB_SIZE = 12
SRC_LEN = 7
TGT_LEN = 4


class AdditionDataset(Dataset):
    """Three-digit addition: operand tokens in, sum digits out."""

    def __init__(self, data_dir: str, split: str,
                 digit_order: str = 'forward', max_pairs: int = None):
        """
        Args:
            data_dir: Directory containing the addition split files
            split: 'train', 'val', or 'test'
            digit_order: 'forward' (most significant digit first) or
                'reversed' (least significant first)
            max_pairs: Keep only the first max_pairs pairs of the split.
                None keeps all
        """
        assert digit_order in ('forward', 'reversed')
        self.digit_order = digit_order
        with open(os.path.join(data_dir, f'{split}.json')) as f:
            pairs = json.load(f)
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        self.pairs = pairs

        src, tgt_in, tgt_out = [], [], []
        for a, b in pairs:
            digits_a = [int(c) for c in f'{a:03d}']
            digits_b = [int(c) for c in f'{b:03d}']
            total = [int(c) for c in f'{a + b:04d}']
            if digit_order == 'reversed':
                total = total[::-1]
            src.append(digits_a + [PLUS] + digits_b)
            tgt_out.append(total)
            tgt_in.append([BOS] + total[:-1])
        self.src = torch.tensor(src, dtype=torch.long).view(-1, SRC_LEN)
        self.tgt_in = torch.tensor(tgt_in, dtype=torch.long).view(-1, TGT_LEN)
        self.tgt_out = torch.tensor(tgt_out,
                                    dtype=torch.long).view(-1, TGT_LEN)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            src: Tensor [7] (long), a2 a1 a0 PLUS b2 b1 b0
            tgt_in: Tensor [4] (long), BOS followed by the first three
                target digits
            tgt_out: Tensor [4] (long), the four target digits in
                digit_order
        """
        return self.src[idx], self.tgt_in[idx], self.tgt_out[idx]
