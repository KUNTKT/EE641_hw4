"""
Offset-retrieval visualization helpers. Provided — do not modify.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_bucket_accuracy(accuracies, bucket_starts, bucket_size, train_len,
                         save_path):
    """
    Accuracy per position bucket, one line per position scheme.

    Args:
        accuracies: Dict {position scheme: list of bucket accuracies}
        bucket_starts: List of the first position of each bucket
        bucket_size: Positions per bucket
        train_len: Training sequence length, marked by a vertical line
        save_path: Output PNG path
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = np.asarray(bucket_starts) + bucket_size / 2
    for name, values in accuracies.items():
        ax.plot(centers, values, marker='o', label=name)
    ax.axvline(train_len, color='0.5', linestyle=':',
               label='training length')
    ax.set_xticks(list(bucket_starts) + [bucket_starts[-1] + bucket_size])
    ax.set_xlabel('Position')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_attention_maps(maps, windows, save_path):
    """
    Attention weights of one sequence, one row per position scheme and
    one column per position window.

    Args:
        maps: Dict {position scheme: list of Tensors or arrays [n, n],
            one per window, query rows and key columns}
        windows: List of (start, end) position ranges, same order
        save_path: Output PNG path
    """
    fig, axes = plt.subplots(len(maps), len(windows),
                             figsize=(4 * len(windows), 3.8 * len(maps)),
                             squeeze=False)
    for row, (name, weights) in enumerate(maps.items()):
        for col, ((start, end), w) in enumerate(zip(windows, weights)):
            if hasattr(w, 'detach'):
                w = w.detach().cpu().numpy()
            ax = axes[row, col]
            ax.imshow(w, cmap='viridis', vmin=0, vmax=1,
                      interpolation='nearest',
                      extent=(start - 0.5, end - 0.5, end - 0.5, start - 0.5))
            ax.set_title(f'{name}, positions {start}-{end - 1}', fontsize=9)
            ax.set_xlabel('Key position')
            ax.set_ylabel('Query position')
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
