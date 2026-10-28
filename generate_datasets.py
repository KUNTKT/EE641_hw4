#!/usr/bin/env python3
"""
Generate the datasets for Problems 1 and 2 and download Problem 3's.

Problem 1: addition. Operand pairs (a, b) drawn uniformly from 0-999,
deduplicated on the unordered pair across all splits. Each split is a
JSON list of [a, b] integer pairs. Tokenization and digit order are the
loader's job.
Problem 2: offset retrieval. Uniform random token sequences over a
16-token vocabulary, stored as int64 arrays [N, T]. Train sequences have
length 64, test sequences length 128. The offset k is fixed in the
handout, not stored in the data.
Problem 3: SMILES strings. A fixed 100,000 / 10,000 subsample of the
MOSES benchmark (MIT license), prepared once on the course side and
served from the course data site. Downloaded here and verified by
checksum, so every student trains on the same strings.

--smoke writes tiny splits for packaging checks. The download is not
reduced: the files are 4 MB.

Usage:
    python generate_datasets.py --seed 641
"""

import argparse
import hashlib
import json
import os
import random
import urllib.request

import numpy as np

ADDITION_SIZES = {'train': 10000, 'val': 2000, 'test': 2000}
ADDITION_SMOKE_SIZES = {'train': 64, 'val': 16, 'test': 16}
MAX_OPERAND = 999

OFFSET_SIZES = {'train': 20000, 'test': 2000}
OFFSET_SMOKE_SIZES = {'train': 64, 'test': 16}
OFFSET_LENGTHS = {'train': 64, 'test': 128}
OFFSET_VOCAB_SIZE = 16

SMILES_URL = 'https://data.usc-ece.com/EE641/HW4/'
SMILES_FILES = {
    'train': ('smiles_train.txt',
              '92578d0437e79f1361e337d6727d6776f36a2596709bfaa4893bac0e30455e73'),
    'val': ('smiles_val.txt',
            'c6968ba787f34dee066de6cab34c0adad97846824d871124b06743d2222f0c6c'),
}


def generate_addition(sizes: dict, rng: random.Random) -> dict:
    """
    Draw unique operand pairs and cut them into splits in order.

    (a, b) and (b, a) count as the same pair, so no sum appears in two
    splits with its operands swapped.

    Returns:
        Dict {split name: list of [a, b] pairs}
    """
    total = sum(sizes.values())
    seen = set()
    pairs = []
    while len(pairs) < total:
        a = rng.randint(0, MAX_OPERAND)
        b = rng.randint(0, MAX_OPERAND)
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append([a, b])

    splits = {}
    start = 0
    for split, count in sizes.items():
        splits[split] = pairs[start:start + count]
        start += count
    return splits


def generate_offset(sizes: dict) -> dict:
    """
    Draw uniform random token sequences for each split.

    Returns:
        Dict {split name: int64 array [N, T]}
    """
    return {split: np.random.randint(0, OFFSET_VOCAB_SIZE,
                                     size=(count, OFFSET_LENGTHS[split]),
                                     dtype=np.int64)
            for split, count in sizes.items()}


def sha256(path: str) -> str:
    """Hex SHA-256 digest of a file's contents."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def download_smiles(out_dir: str) -> None:
    """
    Fetch the SMILES splits into out_dir/{train,val}.txt, skipping files
    already present with the right checksum.
    """
    for split, (name, digest) in SMILES_FILES.items():
        path = os.path.join(out_dir, f'{split}.txt')
        if os.path.exists(path) and sha256(path) == digest:
            print(f'smiles/{split}: present, checksum ok')
            continue
        urllib.request.urlretrieve(SMILES_URL + name, path)
        if sha256(path) != digest:
            os.remove(path)
            raise RuntimeError(f'checksum mismatch for {name}; re-run')
        print(f'smiles/{split}: downloaded -> {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=641)
    parser.add_argument('--output_dir', type=str, default='datasets')
    parser.add_argument('--smoke', action='store_true',
                        help='tiny splits for packaging checks')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    addition_dir = os.path.join(args.output_dir, 'addition')
    os.makedirs(addition_dir, exist_ok=True)
    sizes = ADDITION_SMOKE_SIZES if args.smoke else ADDITION_SIZES
    for split, pairs in generate_addition(sizes, rng).items():
        path = os.path.join(addition_dir, f'{split}.json')
        with open(path, 'w') as f:
            json.dump(pairs, f)
        print(f'addition/{split}: {len(pairs)} pairs -> {path}')

    offset_dir = os.path.join(args.output_dir, 'offset')
    os.makedirs(offset_dir, exist_ok=True)
    sizes = OFFSET_SMOKE_SIZES if args.smoke else OFFSET_SIZES
    for split, tokens in generate_offset(sizes).items():
        path = os.path.join(offset_dir, f'{split}.npz')
        np.savez(path, tokens=tokens)
        print(f'offset/{split}: {tokens.shape[0]}x{tokens.shape[1]} '
              f'tokens -> {path}')

    smiles_dir = os.path.join(args.output_dir, 'smiles')
    os.makedirs(smiles_dir, exist_ok=True)
    download_smiles(smiles_dir)


if __name__ == '__main__':
    main()
