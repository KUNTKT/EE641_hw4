"""
Training script for the addition transformer.

Runs (the handout's experiments):
    python train.py --digit_order forward --train_size 1000
    python train.py --digit_order forward --train_size 2000
    python train.py --digit_order forward --train_size 4000
    python train.py --digit_order reversed --train_size 1000
    python train.py --digit_order reversed --train_size 2000
    python train.py --digit_order reversed --train_size 4000

Each run is named {digit_order}_n{train_size}.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AdditionDataset
from model import Transformer, generate

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_STEPS = 2400
EVAL_EVERY = 50
DATA_ROOT = '../datasets/addition'

TARGET_EXACT_MATCH = 0.95


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_step(model, batch, optimizer, device) -> float:
    """
    One optimizer step on one batch.

    Loss: cross-entropy of the teacher-forced logits against tgt_out,
    averaged over all B x 4 target positions.

    Args:
        batch: (src [B, 7], tgt_in [B, 4], tgt_out [B, 4]) as loaded

    Returns:
        The batch loss
    """
    model.train()
    src, tgt_in, tgt_out = (t.to(device) for t in batch)
    logits = model(src, tgt_in)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           tgt_out.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def validate(model, dataloader, device) -> dict:
    """
    Accuracy under greedy decoding (no teacher forcing).

    Returns:
        Dict with
            exact_match: fraction of sums with all four digits correct
            token_accuracy: fraction of target digits correct
    """
    model.eval()
    exact, correct_tokens, total = 0, 0, 0
    for src, _, tgt_out in dataloader:
        src, tgt_out = src.to(device), tgt_out.to(device)
        correct = generate(model, src) == tgt_out
        exact += correct.all(dim=1).sum().item()
        correct_tokens += correct.sum().item()
        total += src.shape[0]
    return {'exact_match': exact / total,
            'token_accuracy': correct_tokens / (total * tgt_out.shape[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit_order', default='forward',
                        choices=['forward', 'reversed'])
    parser.add_argument('--train_size', type=int, default=4000,
                        help='Number of training pairs used')
    args = parser.parse_args()

    torch.manual_seed(641)
    device = get_device()
    os.makedirs('results', exist_ok=True)
    run = f'{args.digit_order}_n{args.train_size}'

    train_set = AdditionDataset(DATA_ROOT, 'train', args.digit_order,
                                max_pairs=args.train_size)
    val_set = AdditionDataset(DATA_ROOT, 'val', args.digit_order)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1000)

    model = Transformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    log = {'run': run, 'digit_order': args.digit_order,
           'train_size': args.train_size, 'step': [], 'train_loss': [],
           'val_exact_match': [], 'val_token_accuracy': [],
           'steps_to_95': None}
    step, window = 0, []
    while step < NUM_STEPS:
        for batch in train_loader:
            window.append(train_step(model, batch, optimizer, device))
            step += 1
            if step % EVAL_EVERY == 0:
                metrics = validate(model, val_loader, device)
                log['step'].append(step)
                log['train_loss'].append(sum(window) / len(window))
                log['val_exact_match'].append(metrics['exact_match'])
                log['val_token_accuracy'].append(metrics['token_accuracy'])
                if (log['steps_to_95'] is None
                        and metrics['exact_match'] >= TARGET_EXACT_MATCH):
                    log['steps_to_95'] = step
                print(f'{run} step {step:4d}  loss {window[-1]:.4f}  '
                      f'val exact {metrics["exact_match"]:.4f}  '
                      f'token {metrics["token_accuracy"]:.4f}', flush=True)
                window = []
            if step >= NUM_STEPS:
                break

    with open(f'results/training_log_{run}.json', 'w') as f:
        json.dump(log, f, indent=2)
    torch.save(model.state_dict(), f'results/model_{run}.pth')
    print(f'{run}: steps_to_95 = {log["steps_to_95"]}')


if __name__ == '__main__':
    main()
