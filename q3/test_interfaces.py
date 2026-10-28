"""
Interface tests for Problem 3.

These verify shapes, the specified conventions, and the syntax checks on
hand-written strings. They do not require training and do not test
learned behavior.

Run from q3/:  pytest test_interfaces.py
"""

import os

import pytest
import torch
import torch.nn as nn

from dataset import IGNORE_INDEX, SmilesDataset, build_vocab
from evaluate import is_well_formed, max_open_ring_span
from model import Block, DecoderLM, causal_mask
from provided.tokenizer import (BOS_ID, EOS_ID, PAD_ID, SmilesVocab,
                                tokenize)
from sample import generate
from train import lr_at_step

DATA_ROOT = '../datasets/smiles'
TOY_TOKENS = ['C', 'c', 'O', 'N', '(', ')', '1', '2', '=']


def toy_model(norm: str = 'pre') -> DecoderLM:
    return DecoderLM(len(TOY_TOKENS) + 3, d_model=32, num_layers=2,
                     num_heads=4, d_ff=64, max_len=16, dropout=0.1,
                     norm=norm)


class TestCausalMask:

    def test_values(self):
        expected = torch.tensor([[True, False, False],
                                 [True, True, False],
                                 [True, True, True]])
        assert torch.equal(causal_mask(3, torch.device('cpu')), expected)

    def test_dtype_and_shape(self):
        mask = causal_mask(5, torch.device('cpu'))
        assert mask.dtype == torch.bool
        assert mask.shape == (5, 5)


class TestBlock:

    def test_output_shape(self):
        for norm in ('pre', 'post'):
            block = Block(32, 4, 64, norm=norm).eval()
            mask = causal_mask(7, torch.device('cpu'))
            with torch.no_grad():
                out = block(torch.randn(2, 7, 32), mask)
            assert out.shape == (2, 7, 32)

    def test_post_norm_output_normalized(self):
        # Post-norm ends in LayerNorm: zero mean, unit variance per position
        block = Block(32, 4, 64, norm='post').eval()
        mask = causal_mask(7, torch.device('cpu'))
        with torch.no_grad():
            out = block(3 * torch.randn(2, 7, 32) + 1, mask)
        assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 7),
                              atol=1e-5)
        assert torch.allclose(out.var(dim=-1, unbiased=False),
                              torch.ones(2, 7), atol=1e-3)

    def test_causal(self):
        # Changing position 4 must leave positions 0-3 unchanged
        for norm in ('pre', 'post'):
            block = Block(32, 4, 64, norm=norm).eval()
            x = torch.randn(1, 6, 32)
            changed = x.clone()
            changed[0, 4] = torch.randn(32)
            mask = causal_mask(6, torch.device('cpu'))
            with torch.no_grad():
                out_a = block(x, mask)
                out_b = block(changed, mask)
            assert torch.allclose(out_a[0, :4], out_b[0, :4], atol=1e-5)
            assert not torch.allclose(out_a[0, 4], out_b[0, 4])

    def test_no_library_attention(self):
        block = Block(32, 4, 64)
        assert not any(isinstance(m, (nn.MultiheadAttention,
                                      nn.TransformerEncoderLayer))
                       for m in block.modules())


class TestDecoderLM:

    def test_output_shape(self):
        for norm in ('pre', 'post'):
            model = toy_model(norm).eval()
            with torch.no_grad():
                logits = model(torch.randint(0, 12, (3, 10)))
            assert logits.shape == (3, 10, 12)

    def test_final_norm_pre_only(self):
        # Two LayerNorms per block, plus a final one for pre-norm
        pre = sum(isinstance(m, nn.LayerNorm)
                  for m in toy_model('pre').modules())
        post = sum(isinstance(m, nn.LayerNorm)
                   for m in toy_model('post').modules())
        assert pre == 5
        assert post == 4

    def test_causal(self):
        model = toy_model().eval()
        ids = torch.randint(0, 12, (1, 8))
        changed = ids.clone()
        changed[0, 5] = (ids[0, 5] + 1) % 12
        with torch.no_grad():
            out_a = model(ids)
            out_b = model(changed)
        assert torch.allclose(out_a[0, :5], out_b[0, :5], atol=1e-5)


class TestSchedule:

    def test_warmup(self):
        assert lr_at_step(0, 1e-3, 500) == 0.0
        assert lr_at_step(250, 1e-3, 500) == pytest.approx(5e-4)
        assert lr_at_step(500, 1e-3, 500) == pytest.approx(1e-3)
        assert lr_at_step(1200, 1e-3, 500) == pytest.approx(1e-3)

    def test_no_warmup(self):
        assert lr_at_step(0, 1e-2, 0) == pytest.approx(1e-2)
        assert lr_at_step(1, 1e-2, 0) == pytest.approx(1e-2)


class TestWellFormed:

    def test_parentheses(self):
        assert is_well_formed('CC(C)(O)N')
        assert not is_well_formed('CC(C')
        assert not is_well_formed('CC)C(C')  # closes before it opens

    def test_ring_digits(self):
        assert is_well_formed('c1ccccc1')
        assert is_well_formed('C1CC2CC1C2')
        assert not is_well_formed('c1ccccc')
        assert not is_well_formed('C1CC2CC1')

    def test_ring_digit_reused_after_closing(self):
        assert is_well_formed('C1CC1C1CC1')

    def test_bracket_atoms(self):
        assert is_well_formed('c1cc[nH]c1')
        assert not is_well_formed('c1cc[nHc1')


class TestRingSpan:

    def test_spans(self):
        # Tokens c 1 c c c 2 c c c c c 2 c 1: ring 1 spans 13 - 1 = 12,
        # ring 2 spans 11 - 5 = 6
        assert max_open_ring_span('c1ccc2ccccc2c1') == 12
        assert max_open_ring_span('C1CCC1') == 4
        assert max_open_ring_span('C1CCC1C1CC1') == 4

    def test_no_closed_ring(self):
        assert max_open_ring_span('CCO') == 0
        assert max_open_ring_span('C1CCC') == 0


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT),
                    reason='run generate_datasets.py first')
class TestDataset:

    def test_vocab(self):
        vocab = build_vocab(DATA_ROOT)
        assert len(vocab) == 25
        assert vocab.itos[:3] == ['<pad>', '<bos>', '<eos>']

    def test_item_contract(self):
        vocab = build_vocab(DATA_ROOT)
        dataset = SmilesDataset(DATA_ROOT, 'val', vocab, max_len=52)
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (51,) and target_ids.shape == (51,)
        assert input_ids.dtype == torch.long
        assert target_ids.dtype == torch.long

        with open(os.path.join(DATA_ROOT, 'val.txt')) as f:
            encoded = vocab.encode(f.readline().strip())
        n = len(encoded)
        assert input_ids[0] == BOS_ID
        assert input_ids[:n - 1].tolist() == encoded[:-1]
        assert (input_ids[n:] == PAD_ID).all()
        assert target_ids[:n - 1].tolist() == encoded[1:]
        assert target_ids[n - 2] == EOS_ID
        assert (target_ids[n - 1:] == IGNORE_INDEX).all()


class TestGenerate:

    def test_strings_in_vocabulary(self):
        vocab = SmilesVocab(TOY_TOKENS)
        model = toy_model().eval()
        samples = generate(model, vocab, 6, 1.0, torch.device('cpu'),
                           max_len=16)
        assert len(samples) == 6
        for smiles in samples:
            tokens = tokenize(smiles)
            assert ''.join(tokens) == smiles
            assert all(token in TOY_TOKENS for token in tokens)
            assert len(tokens) <= 15


class TestProvidedVocab:

    def test_round_trip(self):
        vocab = SmilesVocab(['C', 'c', 'O', 'N', '(', ')', '1', '[nH]'])
        smiles = 'c1cc[nH]c1C(O)N'
        assert vocab.decode(vocab.encode(smiles)) == smiles


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
