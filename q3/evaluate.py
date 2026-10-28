"""
Evaluation and analysis for the SMILES language model.

Expects the handout's twelve sweep runs (*_s1500) and the main run in
results/. Samples are drawn here with generate rather than read from
sample.py's files. The final model's draw at TEMPERATURE serves both the
last by_step point and the temperature comparison.
"""

import bisect
import glob
import json
import os

import torch

from dataset import build_vocab
from provided.chem import (draw_grid, is_valid, load_canonical_set, novelty,
                           properties, uniqueness, validity)
from provided.tokenizer import tokenize
from provided.visualize import (plot_failure_by_length, plot_failure_by_span,
                                plot_learning_rate_grid,
                                plot_property_distributions,
                                plot_validity_curve)
from sample import TEMPERATURE, generate, load_model
from train import DATA_ROOT, NUM_STEPS, get_device

MAIN_RUN = 'pre_lr0.001_w500_s4000'
SWEEP_STEPS = NUM_STEPS
TRAINS_THRESHOLD = 2.0
EARLY_STEPS = 100

N_EVAL_SAMPLES = 1000
TEMPERATURES = (0.7, 1.0, 1.3)
N_PROPERTY_MOLECULES = 1000
N_GRID = 16

LENGTH_EDGES = [30, 36, 42, 48]
LENGTH_LABELS = ['<=30', '31-36', '37-42', '43-48', '>48']
SPAN_EDGES = [10, 20, 30, 40]
SPAN_LABELS = ['<=10', '11-20', '21-30', '31-40', '>40']


def summarize_run(log: dict) -> dict:
    """
    Sweep summary of one training log.

    Returns:
        Dict with
            run, lr, norm, warmup: from the log's config
            final_val_loss: validation loss at the last step
            trains: final_val_loss < TRAINS_THRESHOLD
            head_grad_norm_first100: mean of the logged head_grad_norm
                values at steps up to EARLY_STEPS (the 50-step window
                means)
            head_grad_norm_max_first100: max of the logged
                head_grad_norm_max values over the same log points
    """
    config = log['config']
    early = [i for i, step in enumerate(log['step']) if step <= EARLY_STEPS]
    final_val_loss = log['val_loss'][-1]
    return {
        'run': config['run'],
        'lr': config['lr'],
        'norm': config['norm'],
        'warmup': config['warmup'],
        'final_val_loss': final_val_loss,
        'trains': bool(final_val_loss < TRAINS_THRESHOLD),
        'head_grad_norm_first100': (sum(log['head_grad_norm'][i]
                                        for i in early) / len(early)),
        'head_grad_norm_max_first100': max(log['head_grad_norm_max'][i]
                                           for i in early),
    }


def is_well_formed(smiles: str) -> bool:
    """
    Syntax check on tokenize(smiles), independent of chemistry. Requires
    that the tokens rejoin to the string (an unclosed bracket atom does
    not tokenize), that no ')' closes more parentheses than are open and
    all are closed at the end, and that every ring-closure digit pairs:
    a digit opens a ring bond, its next occurrence closes it, and a
    closed digit may open a new one.
    """
    tokens = tokenize(smiles)
    if not tokens or ''.join(tokens) != smiles:
        return False
    depth = 0
    open_rings = set()
    for token in tokens:
        if token == '(':
            depth += 1
        elif token == ')':
            depth -= 1
            if depth < 0:
                return False
        elif _is_ring_digit(token):
            open_rings ^= {token}
        elif token.startswith('[') and not token.endswith(']'):
            return False
    return depth == 0 and not open_rings


def _is_ring_digit(token: str) -> bool:
    """Ring-closure label: a single digit or %NN."""
    return token.isdigit() or (token.startswith('%') and len(token) == 3)


def max_open_ring_span(smiles: str) -> int:
    """
    Longest distance, in tokens, from a ring-closure digit to the digit
    that closes it. 0 if no ring closes.
    """
    opened = {}
    longest = 0
    for i, token in enumerate(tokenize(smiles)):
        if not _is_ring_digit(token):
            continue
        if token in opened:
            longest = max(longest, i - opened.pop(token))
        else:
            opened[token] = i
    return longest


def sample_quality(samples: list, train_canonical: set) -> dict:
    """
    Quality fractions for a list of generated strings.

    Returns:
        Dict with well_formed (fraction passing is_well_formed), valid
        (RDKit validity), unique, novel (provided chem definitions)
    """
    return {
        'well_formed': sum(map(is_well_formed, samples)) / len(samples),
        'valid': validity(samples),
        'unique': uniqueness(samples),
        'novel': novelty(samples, train_canonical),
    }


def outcome(smiles: str) -> str:
    """'valid', 'well_formed_invalid', or 'ill_formed'."""
    if not is_well_formed(smiles):
        return 'ill_formed'
    if is_valid(smiles):
        return 'valid'
    return 'well_formed_invalid'


def outcome_buckets(samples: list, values: list, edges: list,
                    labels: list) -> list:
    """
    Outcome counts per bucket of a per-sample value. Bucket i holds
    values in (edges[i - 1], edges[i]], the last bucket values above
    edges[-1]. Each sample is ill_formed (fails is_well_formed), valid
    (RDKit parses it), or well_formed_invalid.

    Args:
        samples: List of SMILES strings
        values: Per-sample value (length or span), same length
        edges: Ascending upper bucket edges
        labels: One label per bucket, len(edges) + 1

    Returns:
        List of dicts in bucket order with label, valid,
        well_formed_invalid, ill_formed (provided/visualize.py)
    """
    buckets = [{'label': label, 'valid': 0, 'well_formed_invalid': 0,
                'ill_formed': 0} for label in labels]
    for smiles, value in zip(samples, values):
        buckets[bisect.bisect_left(edges, value)][outcome(smiles)] += 1
    return buckets


def main():
    torch.manual_seed(641)
    device = get_device()
    out = 'results/visualizations'
    os.makedirs(out, exist_ok=True)
    metrics = {}

    # Sweep: the twelve short runs
    sweep, grid = [], []
    for path in sorted(glob.glob(f'results/training_log_*_s{SWEEP_STEPS}'
                                 '.json')):
        with open(path) as f:
            log = json.load(f)
        summary = summarize_run(log)
        sweep.append(summary)
        grid.append({'lr': summary['lr'], 'norm': summary['norm'],
                     'warmup': summary['warmup'], 'steps': log['step'],
                     'loss': log['train_loss'],
                     'trains': summary['trains']})
        print(f"{summary['run']:24s} val {summary['final_val_loss']:.4f} "
              f"{'trains' if summary['trains'] else 'stalls'}  head grad "
              f"mean {summary['head_grad_norm_first100']:.3f} max "
              f"{summary['head_grad_norm_max_first100']:.3f}")
    sweep.sort(key=lambda r: (r['norm'] != 'pre', r['lr'], r['warmup']))
    metrics['sweep'] = sweep
    if grid:
        plot_learning_rate_grid(grid, os.path.join(out, 'sweep_grid.png'),
                                threshold=TRAINS_THRESHOLD)

    # Main run: samples from each checkpoint and the final model
    main_log_path = f'results/training_log_{MAIN_RUN}.json'
    if os.path.exists(main_log_path):
        with open(main_log_path) as f:
            main_log = json.load(f)
        norm = main_log['config']['norm']
        final_step = main_log['config']['steps']
        vocab = build_vocab(DATA_ROOT)
        train_path = os.path.join(DATA_ROOT, 'train.txt')
        train_canonical = load_canonical_set(train_path)

        checkpoints = {}
        for path in glob.glob(f'results/model_{MAIN_RUN}_step*.pth'):
            step = int(path.rsplit('step', 1)[1].split('.')[0])
            if step < final_step:
                checkpoints[step] = path
        checkpoints[final_step] = f'results/model_{MAIN_RUN}.pth'

        by_step = {'step': [], 'well_formed': [], 'valid': [], 'unique': [],
                   'novel': []}
        final_samples = None
        for step in sorted(checkpoints):
            model = load_model(checkpoints[step], len(vocab), norm, device)
            samples = generate(model, vocab, N_EVAL_SAMPLES, TEMPERATURE,
                               device)
            quality = sample_quality(samples, train_canonical)
            by_step['step'].append(step)
            for key, value in quality.items():
                by_step[key].append(value)
            print(f'step {step}: ' + '  '.join(
                f'{k} {v:.3f}' for k, v in quality.items()))
            if step == final_step:
                final_samples, final_model = samples, model
        plot_validity_curve(by_step, os.path.join(out, 'validity_curve.png'))

        lengths = [len(tokenize(s)) for s in final_samples]
        spans = [max_open_ring_span(s) for s in final_samples]
        by_length = outcome_buckets(final_samples, lengths, LENGTH_EDGES,
                                    LENGTH_LABELS)
        by_span = outcome_buckets(final_samples, spans, SPAN_EDGES,
                                  SPAN_LABELS)
        plot_failure_by_length(by_length,
                               os.path.join(out, 'failure_by_length.png'))
        plot_failure_by_span(by_span, os.path.join(out, 'failure_by_span.png'))

        by_temperature = {}
        for temperature in TEMPERATURES:
            if temperature == TEMPERATURE:
                samples = final_samples
            else:
                samples = generate(final_model, vocab, N_EVAL_SAMPLES,
                                   temperature, device)
            quality = sample_quality(samples, train_canonical)
            by_temperature[f'{temperature:g}'] = quality
            print(f'T={temperature:g}: ' + '  '.join(
                f'{k} {v:.3f}' for k, v in quality.items()))

        examples = {'valid': [], 'well_formed_invalid': [], 'ill_formed': []}
        for smiles in final_samples:
            kind = outcome(smiles)
            if len(examples[kind]) < 10:
                examples[kind].append(smiles)

        metrics['main'] = {'run': MAIN_RUN, 'by_step': by_step,
                           'by_length': by_length, 'by_span': by_span,
                           'by_temperature': by_temperature,
                           'examples': examples}

        valid_samples = [s for s in final_samples if is_valid(s)]
        with open(train_path) as f:
            train_smiles = [line.strip() for line in f if line.strip()]
        train_props = [properties(s)
                       for s in train_smiles[:N_PROPERTY_MOLECULES]]
        sample_props = [properties(s) for s in valid_samples]
        plot_property_distributions(
            sample_props, train_props,
            os.path.join(out, 'property_distributions.png'))
        grid_smiles = valid_samples[:N_GRID]
        draw_grid(grid_smiles, os.path.join(out, 'sample_grid.png'),
                  legends=grid_smiles)

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('wrote results/metrics.json')


if __name__ == '__main__':
    main()
