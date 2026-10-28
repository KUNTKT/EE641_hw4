"""
Training script for the SMILES language model.

Runs (the handout's experiments):
    python train.py --norm pre --lr 1e-3 --warmup 0
    python train.py --norm pre --lr 1e-3 --warmup 500
    python train.py --norm pre --lr 1e-2 --warmup 0
    python train.py --norm pre --lr 1e-2 --warmup 500
    python train.py --norm pre --lr 3e-2 --warmup 0
    python train.py --norm pre --lr 3e-2 --warmup 500
    python train.py --norm post --lr 1e-3 --warmup 0
    python train.py --norm post --lr 1e-3 --warmup 500
    python train.py --norm post --lr 1e-2 --warmup 0
    python train.py --norm post --lr 1e-2 --warmup 500
    python train.py --norm post --lr 3e-2 --warmup 0
    python train.py --norm post --lr 3e-2 --warmup 500
    python train.py --norm pre --lr 1e-3 --warmup 500 --steps 4000 \\
        --checkpoint_every 1000

Each run is named {norm}_lr{lr}_w{warmup}_s{steps}, e.g.
pre_lr0.001_w500_s4000.
Steps count from 1.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SmilesDataset, build_vocab
from model import DecoderLM

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_STEPS = 1500
DATA_ROOT = '../datasets/smiles'

D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
NUM_LAYERS = 4
DROPOUT = 0.1
MAX_LEN = 52
BETAS = (0.9, 0.98)
CLIP_NORM = 1.0
LOG_EVERY = 50
VAL_EVERY = 250


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def lr_at_step(step: int, peak_lr: float, warmup: int) -> float:
    """
    Linear warmup from 0 at step 0 to peak_lr at step warmup, then
    constant. warmup = 0 gives peak_lr at every step.
    """
    if warmup <= 0 or step >= warmup:
        return peak_lr
    return peak_lr * step / warmup


def train_step(model, batch, optimizer, device) -> dict:
    """
    One optimizer step on one batch.

    Loss: cross-entropy of the logits against target_ids, averaged over
    the non-padding targets. Gradient norms are global L2 norms taken
    after the backward pass and before clipping to CLIP_NORM.

    Args:
        batch: (input_ids [B, T], target_ids [B, T]) as loaded

    Returns:
        Dict with
            train_loss: the batch loss
            head_grad_norm: norm of the output head's weight gradient
            total_grad_norm: norm of the gradient of every parameter
    """
    model.train()
    input_ids, target_ids = (t.to(device) for t in batch)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           target_ids.reshape(-1), ignore_index=-100)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    head_grad_norm = model.head.weight.grad.norm().item()
    total_grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), CLIP_NORM).item()
    optimizer.step()
    return {'train_loss': loss.item(), 'head_grad_norm': head_grad_norm,
            'total_grad_norm': total_grad_norm}


@torch.no_grad()
def validate(model, dataloader, device) -> float:
    """Mean cross-entropy over the non-padding targets, batch-averaged."""
    model.eval()
    losses = []
    for input_ids, target_ids in dataloader:
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)
        losses.append(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1),
            ignore_index=-100).item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', default='pre', choices=['pre', 'post'])
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Peak learning rate')
    parser.add_argument('--warmup', type=int, default=0,
                        help='Linear warmup steps (0 for none)')
    parser.add_argument('--steps', type=int, default=NUM_STEPS)
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help='Save weights every N steps (0 for none)')
    args = parser.parse_args()

    torch.manual_seed(641)
    device = get_device()
    os.makedirs('results', exist_ok=True)
    run = f'{args.norm}_lr{args.lr:g}_w{args.warmup}_s{args.steps}'

    vocab = build_vocab(DATA_ROOT)
    train_set = SmilesDataset(DATA_ROOT, 'train', vocab, MAX_LEN)
    val_set = SmilesDataset(DATA_ROOT, 'val', vocab, MAX_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=500)

    model = DecoderLM(len(vocab), D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF,
                      MAX_LEN, DROPOUT, args.norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=BETAS)

    config = {'run': run, 'norm': args.norm, 'lr': args.lr,
              'warmup': args.warmup, 'steps': args.steps,
              'batch_size': BATCH_SIZE, 'd_model': D_MODEL,
              'num_heads': NUM_HEADS, 'd_ff': D_FF,
              'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
              'max_len': MAX_LEN, 'betas': list(BETAS),
              'clip_norm': CLIP_NORM}
    # Logged values are means over each LOG_EVERY-step window, plus the
    # window maxima of the gradient norms
    log = {'config': config, 'step': [], 'train_loss': [], 'lr': [],
           'head_grad_norm': [], 'head_grad_norm_max': [],
           'total_grad_norm': [], 'total_grad_norm_max': [],
           'val_step': [], 'val_loss': []}
    window = []
    step = 0
    while step < args.steps:
        for batch in train_loader:
            step += 1
            lr = lr_at_step(step, args.lr, args.warmup)
            for group in optimizer.param_groups:
                group['lr'] = lr
            window.append(train_step(model, batch, optimizer, device))

            if step % LOG_EVERY == 0 or step == args.steps:
                def mean(key):
                    return sum(w[key] for w in window) / len(window)
                log['step'].append(step)
                log['train_loss'].append(mean('train_loss'))
                log['lr'].append(lr)
                log['head_grad_norm'].append(mean('head_grad_norm'))
                log['head_grad_norm_max'].append(
                    max(w['head_grad_norm'] for w in window))
                log['total_grad_norm'].append(mean('total_grad_norm'))
                log['total_grad_norm_max'].append(
                    max(w['total_grad_norm'] for w in window))
                window = []
            if step % VAL_EVERY == 0 or step == args.steps:
                val_loss = validate(model, val_loader, device)
                log['val_step'].append(step)
                log['val_loss'].append(val_loss)
                print(f'{run} step {step:5d}  lr {lr:.2e}  '
                      f'train {log["train_loss"][-1]:.4f}  '
                      f'val {val_loss:.4f}  '
                      f'head_grad {log["head_grad_norm"][-1]:.3f}',
                      flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                torch.save(model.state_dict(),
                           f'results/model_{run}_step{step}.pth')
            if step >= args.steps:
                break

    log['final_val_loss'] = log['val_loss'][-1]
    with open(f'results/training_log_{run}.json', 'w') as f:
        json.dump(log, f, indent=2)
    torch.save(model.state_dict(), f'results/model_{run}.pth')


if __name__ == '__main__':
    main()
