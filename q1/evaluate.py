"""
Evaluation and analysis for the trained addition transformers.

Scores every run present in results/ on the test split. The head
ablation, pruning curve, and attention maps use the two n4000 runs.
"""

import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import BOS, AdditionDataset
from model import Transformer, generate
from provided.visualize import (SOURCE_LABELS, output_labels,
                                plot_accuracy_vs_size, plot_attention_maps,
                                plot_pruning_curves, plot_val_curves)
from train import DATA_ROOT, get_device, validate

DIGIT_ORDERS = ['forward', 'reversed']
TRAIN_SIZES = [1000, 2000, 4000]
ANALYSIS_SIZE = 4000
EVAL_BATCH_SIZE = 1000


def load_model(path: str, device: torch.device) -> Transformer:
    """Load a trained model in eval mode."""
    model = Transformer()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def remove_head(attention, head: int) -> torch.Tensor:
    """
    Remove one head in place by zeroing its input columns of W_o.

    Args:
        attention: A MultiHeadAttention module
        head: Head index

    Returns:
        The original columns [d_model, d_k], for restore_head
    """
    cols = slice(head * attention.d_k, (head + 1) * attention.d_k)
    with torch.no_grad():
        saved = attention.W_o.weight[:, cols].clone()
        attention.W_o.weight[:, cols] = 0.0
    return saved


def restore_head(attention, head: int, saved: torch.Tensor) -> None:
    """Write back the W_o columns that remove_head returned."""
    cols = slice(head * attention.d_k, (head + 1) * attention.d_k)
    with torch.no_grad():
        attention.W_o.weight[:, cols] = saved


def single_head_ablation(model, dataloader, device) -> list:
    """
    Test exact match with each head removed alone, all others intact.

    Returns:
        List of dicts with 'module', 'head', 'test_exact_match', one per
        head, in Transformer.attention_modules order
    """
    results = []
    for name, module in model.attention_modules():
        for head in range(module.num_heads):
            saved = remove_head(module, head)
            exact = validate(model, dataloader, device)['exact_match']
            restore_head(module, head, saved)
            results.append({'module': name, 'head': head,
                            'test_exact_match': exact})
    return results


def greedy_pruning(model, dataloader, device) -> list:
    """
    Remove heads one at a time, each time the head whose removal costs
    the least test exact match given the heads already removed. Damage is
    recomputed after every removal. Ties go to the earlier head in
    attention_modules order. The model is restored on return.

    Returns:
        List of dicts with 'removed' (e.g. 'dec1_cross.h2') and
        'test_exact_match' after that removal. The first entry has
        'removed' None and the intact model's exact match.
    """
    modules = dict(model.attention_modules())
    remaining = [(name, head) for name, module in model.attention_modules()
                 for head in range(module.num_heads)]
    removed = []  # (name, head, saved columns)
    curve = [{'removed': None,
              'test_exact_match': validate(model, dataloader,
                                           device)['exact_match']}]
    while remaining:
        best, best_exact = None, -1.0
        for name, head in remaining:
            saved = remove_head(modules[name], head)
            exact = validate(model, dataloader, device)['exact_match']
            restore_head(modules[name], head, saved)
            if exact > best_exact:  # strict: ties keep the earlier head
                best, best_exact = (name, head), exact
        name, head = best
        removed.append((name, head, remove_head(modules[name], head)))
        remaining.remove(best)
        curve.append({'removed': f'{name}.h{head}',
                      'test_exact_match': best_exact})
    for name, head, saved in reversed(removed):
        restore_head(modules[name], head, saved)
    return curve


@torch.no_grad()
def mean_attention_maps(model, dataloader, device) -> dict:
    """
    Attention weights averaged over a split. The decoder input is BOS
    followed by the model's own greedy predictions, so the maps show
    the attention used at test time.

    Returns:
        Dict with
            encoder_self: Array [L, h, 7, 7]
            decoder_cross: Array [L, h, 4, 7]
    """
    model.eval()
    encoder_sum, cross_sum, count = 0.0, 0.0, 0
    for src, _, _ in dataloader:
        src = src.to(device)
        predicted = generate(model, src)
        bos = torch.full((src.shape[0], 1), BOS, dtype=torch.long,
                         device=device)
        tgt_in = torch.cat([bos, predicted[:, :-1]], dim=1)
        weights = model.attention(src, tgt_in)
        encoder_sum = encoder_sum + torch.stack(
            weights['encoder_self']).sum(dim=1)
        cross_sum = cross_sum + torch.stack(
            weights['decoder_cross']).sum(dim=1)
        count += src.shape[0]
    return {'encoder_self': (encoder_sum / count).cpu().numpy(),
            'decoder_cross': (cross_sum / count).cpu().numpy()}


def main():
    device = get_device()
    out = 'results/visualizations'
    os.makedirs(out, exist_ok=True)

    metrics, logs, by_size = {}, {}, {order: {} for order in DIGIT_ORDERS}
    loaders = {}
    for order in DIGIT_ORDERS:
        test_set = AdditionDataset(DATA_ROOT, 'test', order)
        loaders[order] = DataLoader(test_set, batch_size=EVAL_BATCH_SIZE)
        for size in TRAIN_SIZES:
            run = f'{order}_n{size}'
            log_path = f'results/training_log_{run}.json'
            model_path = f'results/model_{run}.pth'
            if not (os.path.exists(log_path) and os.path.exists(model_path)):
                continue
            with open(log_path) as f:
                logs[run] = json.load(f)
            model = load_model(model_path, device)
            test = validate(model, loaders[order], device)
            metrics[run] = {'test_exact_match': test['exact_match'],
                            'test_token_accuracy': test['token_accuracy'],
                            'steps_to_95': logs[run]['steps_to_95']}
            by_size[order][size] = test['exact_match']
            print(f'{run}: test exact {test["exact_match"]:.4f}  token '
                  f'{test["token_accuracy"]:.4f}  steps_to_95 '
                  f'{logs[run]["steps_to_95"]}')

    plot_val_curves(logs, os.path.join(out, 'val_exact_match.png'))
    plot_accuracy_vs_size({o: s for o, s in by_size.items() if s},
                          os.path.join(out, 'test_accuracy_vs_size.png'))

    metrics['ablation'], metrics['pruning'] = {}, {}
    metrics['attention'] = {}
    curves = {}
    for order in DIGIT_ORDERS:
        run = f'{order}_n{ANALYSIS_SIZE}'
        if run not in metrics:
            continue
        model = load_model(f'results/model_{run}.pth', device)
        loader = loaders[order]

        metrics['ablation'][run] = single_head_ablation(model, loader,
                                                        device)
        curves[run] = greedy_pruning(model, loader, device)
        metrics['pruning'][run] = curves[run]

        maps = mean_attention_maps(model, loader, device)
        metrics['attention'][run] = {key: value.round(4).tolist()
                                     for key, value in maps.items()}
        plot_attention_maps(
            maps['decoder_cross'], output_labels(order), SOURCE_LABELS,
            os.path.join(out, f'cross_attention_{run}.png'),
            title=f'Decoder cross-attention, {run}')
        plot_attention_maps(
            maps['encoder_self'], SOURCE_LABELS, SOURCE_LABELS,
            os.path.join(out, f'encoder_self_attention_{run}.png'),
            title=f'Encoder self-attention, {run}')
        print(f'{run}: ablation and pruning done')

    if curves:
        plot_pruning_curves(curves, os.path.join(out, 'pruning_curves.png'))

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('wrote results/metrics.json')


if __name__ == '__main__':
    main()
