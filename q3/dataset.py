"""
Data loading for the SMILES language model.

Each string is encoded as [BOS, tokens, EOS] and padded with PAD to
max_len ids. The input is every id but the last and the target is every
id but the first, so position t predicts id t + 1. Targets at padding
carry IGNORE_INDEX.
"""

import os

import torch
from torch.utils.data import Dataset

from provided.tokenizer import PAD_ID, SmilesVocab, tokenize

IGNORE_INDEX = -100


def build_vocab(data_dir: str) -> SmilesVocab:
    """Vocabulary of every token in the training file (train.txt)."""
    tokens = set()
    with open(os.path.join(data_dir, 'train.txt')) as f:
        for line in f:
            if line.strip():
                tokens.update(tokenize(line.strip()))
    return SmilesVocab(tokens)


class SmilesDataset(Dataset):
    """SMILES strings, one per line, as next-token prediction pairs."""

    def __init__(self, data_dir: str, split: str, vocab: SmilesVocab,
                 max_len: int):
        """
        Args:
            data_dir: Directory containing train.txt and val.txt
            split: 'train' or 'val'
            vocab: Vocabulary built from the training file
            max_len: Padded length of an encoded string, in ids
        """
        with open(os.path.join(data_dir, f'{split}.txt')) as f:
            self.smiles = [line.strip() for line in f if line.strip()]
        ids = torch.full((len(self.smiles), max_len), PAD_ID,
                         dtype=torch.long)
        for i, smiles in enumerate(self.smiles):
            encoded = vocab.encode(smiles)[:max_len]
            ids[i, :len(encoded)] = torch.tensor(encoded)
        self.inputs = ids[:, :-1].clone()
        self.targets = ids[:, 1:].clone()
        self.targets[self.targets == PAD_ID] = IGNORE_INDEX

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            input_ids: Tensor [max_len - 1] (long), BOS first, PAD after
                EOS
            target_ids: Tensor [max_len - 1] (long), input_ids shifted
                left by one, IGNORE_INDEX where the target is PAD
        """
        return self.inputs[idx], self.targets[idx]
