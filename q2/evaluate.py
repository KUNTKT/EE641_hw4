"""
Length-extrapolation evaluation for the offset-retrieval model.

Expects one or more of the handout's four training runs in results/
(model_{position}.pth). Test sequences have 128 positions: 0 to 63 is
the training range, 64 to 127 lies beyond it.
"""

import json
import math
import os

import torch
from torch.utils.data import DataLoader

from dataset import IGNORE_INDEX, OffsetDataset
from positional import (apply_rotary, score_shift_invariance, shift_rotation,
                        sinusoidal_table)
from provided.model import POSITIONS, OffsetModel
from provided.visualize import plot_attention_maps, plot_bucket_accuracy
from train import (BATCH_SIZE, D_MODEL, DATA_ROOT, K, MAX_LEN, TRAIN_LEN,
                   VOCAB_SIZE, get_device)

BUCKET_SIZE = 16
MAP_POSITIONS = ('sinusoidal', 'rotary')
MAP_WINDOWS = [(0, 32), (96, 128)]
SHIFT_CASES = [(0, 1), (5, 3), (10, 7), (37, 20), (60, 64)]
SCORE_SHIFTS = [1, 16, 64]


def load_model(path: str, position: str,
               device: torch.device) -> OffsetModel:
    """Load a trained model in eval mode."""
    model = OffsetModel(VOCAB_SIZE, D_MODEL, MAX_LEN, position)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def position_counts(model, dataloader, device) -> tuple:
    """
    Correct predictions and scored targets at each position, summed over
    the dataset.

    Returns:
        correct: Tensor [T] (long)
        counts: Tensor [T] (long), zero where no position has a target
    """
    model.eval()
    correct, counts = None, None
    for tokens, targets in dataloader:
        tokens, targets = tokens.to(device), targets.to(device)
        logits, _ = model(tokens)
        scored = targets != IGNORE_INDEX
        hits = (logits.argmax(dim=-1) == targets) & scored
        if correct is None:
            correct = torch.zeros(tokens.shape[1], dtype=torch.long)
            counts = torch.zeros(tokens.shape[1], dtype=torch.long)
        correct += hits.sum(dim=0).cpu()
        counts += scored.sum(dim=0).cpu()
    return correct, counts


def range_accuracy(correct: torch.Tensor, counts: torch.Tensor, start: int,
                   end: int) -> float:
    """Accuracy over positions start to end - 1."""
    total = counts[start:end].sum().item()
    return correct[start:end].sum().item() / total if total else float('nan')


def bucket_accuracies(correct: torch.Tensor, counts: torch.Tensor,
                      start: int, end: int,
                      size: int = BUCKET_SIZE) -> list:
    """
    Accuracy in consecutive buckets of size positions from start to end.

    Returns:
        List [(end - start) / size] of accuracies
    """
    return [range_accuracy(correct, counts, s, s + size)
            for s in range(start, end, size)]


@torch.no_grad()
def attention_windows(model, tokens: torch.Tensor, device,
                      windows: list = MAP_WINDOWS) -> list:
    """
    Attention weights of one sequence, cut to square windows on the
    diagonal.

    Args:
        tokens: Tensor [T] (long)
        windows: List of (start, end) position ranges

    Returns:
        List of Tensors [end - start, end - start], query rows and key
        columns start to end - 1
    """
    model.eval()
    _, weights = model(tokens[None].to(device))
    weights = weights[0].cpu()
    return [weights[start:end, start:end] for start, end in windows]


def pairwise_rotary(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Reference rotary encoding: each pair (2i, 2i+1) at position p
    multiplied by its own 2x2 rotation matrix by p * 10000^(-2i / d).

    Args:
        x: Tensor [T, d]
        positions: Tensor [T]

    Returns:
        Tensor [T, d]
    """
    T, d = x.shape
    out = torch.empty_like(x)
    for t in range(T):
        p = float(positions[t])
        for i in range(d // 2):
            angle = p * 10000.0 ** (-2 * i / d)
            c, s = math.cos(angle), math.sin(angle)
            rotation = torch.tensor([[c, -s], [s, c]], dtype=x.dtype)
            out[t, 2 * i:2 * i + 2] = rotation @ x[t, 2 * i:2 * i + 2]
    return out


def encoding_checks() -> dict:
    """
    Model-free checks of the position encodings.

    Returns:
        Dict with
            shift_rotation_max_error: max |table[pos + shift]
                - shift_rotation(shift) @ table[pos]| over SHIFT_CASES
            rotary_score_shift_error: max score_shift_invariance over
                SCORE_SHIFTS, rotary, random q and k
            sinusoidal_score_shift_error: the same for sinusoidal
            rotary_vs_pairwise_error: max |apply_rotary - pairwise_rotary|
                over random vectors at positions 0 to MAX_LEN - 1
    """
    table = sinusoidal_table(MAX_LEN + 64, D_MODEL)
    shift_error = max(
        (table[pos + shift] - shift_rotation(shift, D_MODEL)
         @ table[pos]).abs().max().item()
        for pos, shift in SHIFT_CASES)

    q, k = torch.randn(D_MODEL), torch.randn(D_MODEL)
    rotary_error = max(score_shift_invariance(q, k, s, 'rotary')
                       for s in SCORE_SHIFTS)
    sinusoidal_error = max(score_shift_invariance(q, k, s, 'sinusoidal')
                           for s in SCORE_SHIFTS)

    x = torch.randn(MAX_LEN, D_MODEL)
    positions = torch.arange(MAX_LEN)
    pairwise_error = (apply_rotary(x, positions)
                      - pairwise_rotary(x, positions)).abs().max().item()
    return {'shift_rotation_max_error': shift_error,
            'rotary_score_shift_error': rotary_error,
            'sinusoidal_score_shift_error': sinusoidal_error,
            'rotary_vs_pairwise_error': pairwise_error}


def main():
    torch.manual_seed(641)
    device = get_device()
    out = 'results/visualizations'
    os.makedirs(out, exist_ok=True)

    test_set = OffsetDataset(DATA_ROOT, 'test', K)
    loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    seq_len = test_set[0][0].shape[0]

    metrics, buckets, maps = {}, {}, {}
    for position in POSITIONS:
        path = f'results/model_{position}.pth'
        if not os.path.exists(path):
            continue
        model = load_model(path, position, device)
        correct, counts = position_counts(model, loader, device)
        metrics[position] = {
            'in_range': range_accuracy(correct, counts, 0, TRAIN_LEN),
            'beyond': range_accuracy(correct, counts, TRAIN_LEN, seq_len),
            'beyond_buckets': bucket_accuracies(correct, counts, TRAIN_LEN,
                                                seq_len),
            'all_buckets': bucket_accuracies(correct, counts, 0, seq_len),
        }
        buckets[position] = metrics[position]['all_buckets']
        if position in MAP_POSITIONS:
            maps[position] = attention_windows(model, test_set[0][0], device)
        print(f'{position}: in_range {metrics[position]["in_range"]:.4f}  '
              f'beyond {metrics[position]["beyond"]:.4f}')

    metrics['checks'] = encoding_checks()
    for name, value in metrics['checks'].items():
        print(f'{name}: {value:.3e}')

    if buckets:
        # The first bucket holds positions 7-15 only: 0-6 have no target
        plot_bucket_accuracy(buckets, list(range(0, seq_len, BUCKET_SIZE)),
                             BUCKET_SIZE, TRAIN_LEN,
                             os.path.join(out, 'bucket_accuracy.png'))
    if maps:
        plot_attention_maps(maps, MAP_WINDOWS,
                            os.path.join(out, 'attention_maps.png'))

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('wrote results/metrics.json')


if __name__ == '__main__':
    main()
