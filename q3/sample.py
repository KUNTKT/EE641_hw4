"""
Autoregressive sampling from a trained SMILES language model.

Runs (after training the main run):
    python sample.py --run pre_lr0.001_w500_s4000

Writes results/samples_{run}_T{temperature}.txt, one SMILES per line.
"""

import argparse
import json
import os

import torch

from dataset import build_vocab
from model import DecoderLM
from provided.tokenizer import BOS_ID, EOS_ID, PAD_ID, SmilesVocab
from train import (D_FF, D_MODEL, DATA_ROOT, DROPOUT, MAX_LEN, NUM_HEADS,
                   NUM_LAYERS, get_device)

N_SAMPLES = 1000
TEMPERATURE = 1.0


def load_model(path: str, vocab_size: int, norm: str,
               device: torch.device) -> DecoderLM:
    """Load trained weights in eval mode."""
    model = DecoderLM(vocab_size, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF,
                      MAX_LEN, DROPOUT, norm)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def generate(model, vocab: SmilesVocab, n: int, temperature: float,
             device: torch.device, max_len: int = MAX_LEN) -> list:
    """
    Sample n strings in one batch, starting from BOS. Each step draws the
    next id from softmax(logits / temperature) at the last position. A
    sequence is finished once it emits EOS, and sampling stops when every
    sequence is finished or holds max_len ids. Call with the model in
    eval mode.

    Returns:
        List of n SMILES strings (vocab.decode of each sequence)
    """
    ids = torch.full((n, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    while ids.shape[1] < max_len and not finished.all():
        logits = model(ids)[:, -1] / temperature
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)[:, 0]
        next_id = torch.where(finished, torch.full_like(next_id, PAD_ID),
                              next_id)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        finished |= next_id == EOS_ID
    return [vocab.decode(row) for row in ids.cpu()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True,
                        help='Run name, e.g. pre_lr0.001_w500_s4000')
    parser.add_argument('--n', type=int, default=N_SAMPLES)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    args = parser.parse_args()

    torch.manual_seed(641)
    device = get_device()
    os.makedirs('results', exist_ok=True)

    vocab = build_vocab(DATA_ROOT)
    with open(f'results/training_log_{args.run}.json') as f:
        norm = json.load(f)['config']['norm']
    model = load_model(f'results/model_{args.run}.pth', len(vocab), norm,
                       device)
    samples = generate(model, vocab, args.n, args.temperature, device)
    path = f'results/samples_{args.run}_T{args.temperature:g}.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(samples) + '\n')
    print(f'{len(samples)} samples -> {path}')


if __name__ == '__main__':
    main()
