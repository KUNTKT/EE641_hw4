"""
Training script for the offset-retrieval model.

Runs (the handout's experiments):
    python train.py --position none
    python train.py --position learned
    python train.py --position sinusoidal
    python train.py --position rotary
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import IGNORE_INDEX, OffsetDataset
from provided.model import POSITIONS, OffsetModel

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
DATA_ROOT = '../datasets/offset'

K = 7
D_MODEL = 64
VOCAB_SIZE = 16
MAX_LEN = 128
TRAIN_LEN = 64


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_epoch(model, dataloader, optimizer, device) -> dict:
    """
    Train for one epoch.

    Loss: cross-entropy over the positions that have a target
    (IGNORE_INDEX excluded), averaged over those positions.

    Returns:
        Dict of epoch means: loss, accuracy (over positions with a
        target)
    """
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for tokens, targets in dataloader:
        tokens, targets = tokens.to(device), targets.to(device)
        logits, _ = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1),
                               ignore_index=IGNORE_INDEX)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scored = targets != IGNORE_INDEX
        n = scored.sum().item()
        total_loss += loss.item() * n
        correct += ((logits.argmax(-1) == targets) & scored).sum().item()
        count += n
    return {'loss': total_loss / count, 'accuracy': correct / count}


@torch.no_grad()
def validate(model, dataloader, device, length: int = TRAIN_LEN) -> dict:
    """
    Mean loss and accuracy on the first length positions of each
    sequence, the training range.
    """
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for tokens, targets in dataloader:
        tokens = tokens[:, :length].to(device)
        targets = targets[:, :length].to(device)
        logits, _ = model(tokens)
        total_loss += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction='sum').item()
        scored = targets != IGNORE_INDEX
        correct += ((logits.argmax(-1) == targets) & scored).sum().item()
        count += scored.sum().item()
    return {'loss': total_loss / count, 'accuracy': correct / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--position', required=True, choices=POSITIONS,
                        help='Position scheme, also the run name')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    torch.manual_seed(641)
    device = get_device()
    os.makedirs('results', exist_ok=True)

    train_set = OffsetDataset(DATA_ROOT, 'train', K)
    test_set = OffsetDataset(DATA_ROOT, 'test', K)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = OffsetModel(VOCAB_SIZE, D_MODEL, MAX_LEN,
                        args.position).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    log = {'position': args.position, 'epoch': [], 'train_loss': [],
           'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        test_metrics = validate(model, test_loader, device, TRAIN_LEN)
        log['epoch'].append(epoch)
        log['train_loss'].append(train_metrics['loss'])
        log['train_accuracy'].append(train_metrics['accuracy'])
        log['test_loss'].append(test_metrics['loss'])
        log['test_accuracy'].append(test_metrics['accuracy'])
        print(f'{args.position} epoch {epoch}  '
              f'train loss {train_metrics["loss"]:.4f} '
              f'acc {train_metrics["accuracy"]:.4f}  '
              f'test[0:64] loss {test_metrics["loss"]:.4f} '
              f'acc {test_metrics["accuracy"]:.4f}', flush=True)

    with open(f'results/training_log_{args.position}.json', 'w') as f:
        json.dump(log, f, indent=2)
    torch.save(model.state_dict(), f'results/model_{args.position}.pth')


if __name__ == '__main__':
    main()
