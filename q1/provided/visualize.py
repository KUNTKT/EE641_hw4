"""
Addition transformer visualization helpers. Provided — do not modify.
"""

import matplotlib.pyplot as plt
import numpy as np

SOURCE_LABELS = ['a2', 'a1', 'a0', '+', 'b2', 'b1', 'b0']


def output_labels(digit_order):
    """Place value of each output step: 's3' is the thousands digit."""
    labels = ['s3', 's2', 's1', 's0']
    return labels if digit_order == 'forward' else labels[::-1]


def plot_val_curves(logs, save_path):
    """
    Validation exact match against training step, one line per run.

    Args:
        logs: Dict {run name: training log dict with 'step' and
            'val_exact_match'}
        save_path: Output PNG path
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, log in sorted(logs.items()):
        style = '-' if run.startswith('reversed') else '--'
        ax.plot(log['step'], log['val_exact_match'], style, label=run)
    ax.axhline(0.95, color='0.5', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Validation exact match')
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_accuracy_vs_size(results, save_path):
    """
    Test exact match against training-set size, one line per digit order.

    Args:
        results: Dict {digit order: Dict {train size: exact match}}
        save_path: Output PNG path
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for order, by_size in sorted(results.items()):
        sizes = sorted(by_size)
        ax.plot(sizes, [by_size[n] for n in sizes], 'o-', label=order)
    ax.set_xscale('log')
    ax.set_xticks(sorted({n for by_size in results.values()
                          for n in by_size}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel('Training pairs')
    ax.set_ylabel('Test exact match')
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_attention_maps(maps, row_labels, col_labels, save_path,
                        title=None):
    """
    Grid of attention maps: one row per layer, one column per head.

    Args:
        maps: Array [L, h, T_q, T_k], rows sum to 1
        row_labels: T_q query labels
        col_labels: T_k key labels
        save_path: Output PNG path
        title: Optional figure title
    """
    maps = np.asarray(maps)
    n_layers, n_heads = maps.shape[:2]
    fig, axes = plt.subplots(n_layers, n_heads, squeeze=False,
                             figsize=(n_heads * 2.4, n_layers * 2.2))
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            ax.imshow(maps[layer, head], cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=7)
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.set_title(f'layer {layer}, head {head}', fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_pruning_curves(curves, save_path):
    """
    Test exact match against number of heads removed.

    Args:
        curves: Dict {run name: pruning list of dicts with
            'test_exact_match'}, the first entry being no heads removed
        save_path: Output PNG path
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, curve in sorted(curves.items()):
        ax.plot(np.arange(len(curve)),
                [entry['test_exact_match'] for entry in curve], 'o-',
                markersize=3, label=run)
    ax.set_xlabel('Heads removed')
    ax.set_ylabel('Test exact match')
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
