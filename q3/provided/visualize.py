"""
SMILES language model visualization helpers. Provided — do not modify.
"""

import matplotlib.pyplot as plt
import numpy as np

OUTCOMES = [
    ('valid', 'valid', 'tab:green'),
    ('well_formed_invalid', 'well-formed, invalid', 'tab:orange'),
    ('ill_formed', 'ill-formed', 'tab:red'),
]
PROPERTIES = [
    ('qed', 'QED'),
    ('logp', 'Crippen logP'),
    ('mol_weight', 'Molecular weight (Da)'),
]


def plot_learning_rate_grid(results, save_path, threshold=None):
    """
    Training loss curves of the stability sweep, one panel per
    (norm, lr), one curve per warmup setting.

    Args:
        results: List of run dicts, each with
            lr: float, peak learning rate
            norm: 'pre' or 'post'
            warmup: int, warmup steps (0 for none)
            steps: list of int, logged steps
            loss: list of float, training loss at those steps
            trains: bool, the run's trains/stalls verdict
        save_path: Output PNG path
        threshold: Optional loss threshold drawn as a dashed line
    """
    norms = sorted({r['norm'] for r in results}, reverse=True)
    lrs = sorted({r['lr'] for r in results})
    fig, axes = plt.subplots(len(norms), len(lrs),
                             figsize=(3.5 * len(lrs), 2.8 * len(norms)),
                             sharex=True, sharey=True, squeeze=False)
    for i, norm in enumerate(norms):
        for j, lr in enumerate(lrs):
            ax = axes[i, j]
            runs = sorted((r for r in results
                           if r['norm'] == norm and r['lr'] == lr),
                          key=lambda r: r['warmup'])
            for r in runs:
                verdict = 'trains' if r['trains'] else 'stalls'
                ax.plot(r['steps'], r['loss'],
                        label=f"warmup {r['warmup']}: {verdict}")
            if threshold is not None:
                ax.axhline(threshold, color='0.5', linestyle='--',
                           linewidth=1)
            ax.set_title(f'{norm}-norm, lr {lr:g}', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            if i == len(norms) - 1:
                ax.set_xlabel('Step')
            if j == 0:
                ax.set_ylabel('Training loss')
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_validity_curve(log, save_path):
    """
    Well-formed and RDKit-valid fractions of samples over training.

    Args:
        log: Dict with
            step: list of int, steps at which samples were drawn
            well_formed: list of float in [0, 1]
            valid: list of float in [0, 1], RDKit validity
        save_path: Output PNG path
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(log['step'], log['well_formed'], marker='o', markersize=3,
            label='well-formed')
    ax.plot(log['step'], log['valid'], marker='o', markersize=3,
            label='valid (RDKit)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction of samples')
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _plot_outcome_bars(buckets, xlabel, save_path):
    """Stacked outcome fractions per bucket, sample count above each bar."""
    labels = [b['label'] for b in buckets]
    counts = np.array([[b[key] for key, _, _ in OUTCOMES] for b in buckets],
                      dtype=float)
    totals = counts.sum(axis=1)
    fractions = counts / np.maximum(totals, 1)[:, None]

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(buckets)), 4.5))
    x = np.arange(len(buckets))
    bottom = np.zeros(len(buckets))
    for k, (_, name, color) in enumerate(OUTCOMES):
        ax.bar(x, fractions[:, k], bottom=bottom, color=color, label=name)
        bottom += fractions[:, k]
    for xi, n in zip(x, totals):
        ax.text(xi, 1.01, f'n={int(n)}', ha='center', va='bottom',
                fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Fraction of samples')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_failure_by_length(buckets, save_path):
    """
    Sample outcomes by generated length.

    Args:
        buckets: List of dicts in bucket order, each with
            label: str, e.g. '20-29'
            valid: int, samples RDKit parses
            well_formed_invalid: int, well-formed samples RDKit rejects
            ill_formed: int, samples failing the well-formedness check
        save_path: Output PNG path
    """
    _plot_outcome_bars(buckets, 'Length (tokens)', save_path)


def plot_failure_by_span(buckets, save_path):
    """
    Sample outcomes by longest ring-closure span, in tokens.

    Args:
        buckets: List of dicts in bucket order, same keys as
            plot_failure_by_length
        save_path: Output PNG path
    """
    _plot_outcome_bars(buckets, 'Longest ring-closure span (tokens)',
                       save_path)


def plot_property_distributions(sample_props, train_props, save_path):
    """
    Histograms of QED, logP, and molecular weight, samples against the
    training set, each normalized to a density.

    Args:
        sample_props: List of chem.properties dicts (keys qed, logp,
            mol_weight) for valid samples
        train_props: Same, for training molecules
        save_path: Output PNG path
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, name) in zip(axes, PROPERTIES):
        samples = np.array([p[key] for p in sample_props])
        train = np.array([p[key] for p in train_props])
        bins = np.histogram_bin_edges(np.concatenate([samples, train]),
                                      bins=40)
        ax.hist(train, bins=bins, density=True, alpha=0.5,
                label='training set')
        ax.hist(samples, bins=bins, density=True, alpha=0.5,
                label='samples')
        ax.set_xlabel(name)
        ax.set_yticks([])
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=8)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
