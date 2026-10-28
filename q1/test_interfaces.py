"""
Interface tests for Problem 1.

These verify shapes, ranges, hand-computed attention cases, and the
specified conventions. They do not require training and do not test
learned behavior.

Run from q1/:  pytest test_interfaces.py
"""

import os

import pytest
import torch

from attention import (MultiHeadAttention, causal_mask,
                       scaled_dot_product_attention)
from dataset import BOS, PLUS, VOCAB_SIZE, AdditionDataset
from model import DecoderLayer, EncoderLayer, Transformer, generate

DATA_ROOT = '../datasets/addition'


class TestScaledDotProductAttention:

    def test_equal_scores_uniform(self):
        # Zero queries give equal scores, so the output is the mean value
        q = torch.zeros(1, 1, 2, 4)
        k = torch.randn(1, 1, 3, 4)
        v = torch.tensor([[[[1.0, 0, 0, 0], [0, 2.0, 0, 0],
                            [0, 0, 3.0, 0]]]])
        output, weights = scaled_dot_product_attention(q, k, v)
        assert torch.allclose(weights, torch.full((1, 1, 2, 3), 1 / 3))
        assert torch.allclose(output[0, 0, 0],
                              torch.tensor([1 / 3, 2 / 3, 1.0, 0.0]))

    def test_one_hot_key_selects_value(self):
        # Score 100 / sqrt(4) = 50 on key 1 and 0 elsewhere
        q = torch.tensor([[[[0.0, 100.0, 0.0, 0.0]]]])
        k = torch.eye(4)[:3].view(1, 1, 3, 4)
        v = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
        output, weights = scaled_dot_product_attention(q, k, v)
        assert torch.allclose(weights, torch.tensor([[[[0.0, 1.0, 0.0]]]]),
                              atol=1e-6)
        assert torch.allclose(output, v[:, :, 1:2], atol=1e-4)

    def test_scaling_by_sqrt_dk(self):
        # Scores 2 / sqrt(4) = 1 and 0 give weights e / (e + 1), 1 / (e + 1)
        q = torch.tensor([[[[2.0, 0.0, 0.0, 0.0]]]])
        k = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])
        v = torch.zeros(1, 1, 2, 4)
        _, weights = scaled_dot_product_attention(q, k, v)
        e = torch.exp(torch.tensor(1.0))
        assert torch.allclose(weights[0, 0, 0],
                              torch.stack([e / (e + 1), 1 / (e + 1)]))

    def test_masked_positions_zero_weight(self):
        q = torch.zeros(1, 1, 1, 4)
        k = torch.randn(1, 1, 3, 4)
        v = torch.randn(1, 1, 3, 4)
        mask = torch.tensor([True, False, True])
        output, weights = scaled_dot_product_attention(q, k, v, mask)
        assert torch.equal(weights[0, 0, 0, 1], torch.tensor(0.0))
        assert torch.allclose(weights[0, 0, 0], torch.tensor([0.5, 0, 0.5]))
        assert torch.allclose(output[0, 0, 0], (v[0, 0, 0] + v[0, 0, 2]) / 2)


class TestCausalMask:

    def test_triangle(self):
        expected = torch.tensor([[True, False, False],
                                 [True, True, False],
                                 [True, True, True]])
        mask = causal_mask(3, torch.device('cpu'))
        assert mask.dtype == torch.bool
        assert torch.equal(mask, expected)

    def test_shape(self):
        assert causal_mask(4, torch.device('cpu')).shape == (4, 4)


class TestMultiHeadAttention:

    def test_shapes_and_rows(self):
        mha = MultiHeadAttention(d_model=128, num_heads=4).eval()
        query = torch.randn(2, 4, 128)
        memory = torch.randn(2, 7, 128)
        with torch.no_grad():
            output, weights = mha(query, memory, memory)
        assert output.shape == (2, 4, 128)
        assert weights.shape == (2, 4, 4, 7)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 4))

    def test_projections(self):
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        for name in ('W_q', 'W_k', 'W_v', 'W_o'):
            layer = getattr(mha, name)
            assert isinstance(layer, torch.nn.Linear)
            assert layer.weight.shape == (128, 128)

    def test_causal_mask_applied(self):
        mha = MultiHeadAttention(d_model=128, num_heads=4).eval()
        x = torch.randn(2, 4, 128)
        with torch.no_grad():
            _, weights = mha(x, x, x, causal_mask(4, torch.device('cpu')))
        upper = torch.ones(4, 4, dtype=torch.bool).triu(diagonal=1)
        assert torch.all(weights[..., upper] == 0)

    def test_zeroing_head_columns(self):
        # Head 1 enters W_o through input columns 32-63
        mha = MultiHeadAttention(d_model=128, num_heads=4).eval()
        x = torch.randn(2, 5, 128)
        with torch.no_grad():
            before, _ = mha(x, x, x)
            mha.W_o.weight[:, 32:64] = 0.0
            after, _ = mha(x, x, x)
        assert after.shape == before.shape
        assert not torch.allclose(after, before)


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT),
                    reason='run generate_datasets.py first')
class TestDataset:

    def test_item_contract(self):
        dataset = AdditionDataset(DATA_ROOT, 'val')
        src, tgt_in, tgt_out = dataset[0]
        assert src.shape == (7,) and src.dtype == torch.long
        assert tgt_in.shape == (4,) and tgt_in.dtype == torch.long
        assert tgt_out.shape == (4,) and tgt_out.dtype == torch.long
        assert src[3] == PLUS
        assert torch.all(torch.cat([src[:3], src[4:]]) < 10)
        assert torch.all(tgt_out < 10)

    def test_sum_and_shift(self):
        src, tgt_in, tgt_out = AdditionDataset(DATA_ROOT, 'val')[0]
        a = int(''.join(str(d) for d in src[:3].tolist()))
        b = int(''.join(str(d) for d in src[4:].tolist()))
        assert int(''.join(str(d) for d in tgt_out.tolist())) == a + b
        assert tgt_in[0] == BOS
        assert torch.equal(tgt_in[1:], tgt_out[:-1])

    def test_reversed_order(self):
        forward = AdditionDataset(DATA_ROOT, 'val', 'forward')[0]
        reverse = AdditionDataset(DATA_ROOT, 'val', 'reversed')[0]
        assert torch.equal(forward[0], reverse[0])
        assert torch.equal(reverse[2], forward[2].flip(0))
        assert reverse[1][0] == BOS
        assert torch.equal(reverse[1][1:], reverse[2][:-1])

    def test_max_pairs(self):
        full = AdditionDataset(DATA_ROOT, 'train')
        subset = AdditionDataset(DATA_ROOT, 'train', max_pairs=100)
        assert len(subset) == 100
        assert torch.equal(subset[99][0], full[99][0])


class TestLayers:

    def test_encoder_layer(self):
        layer = EncoderLayer(128, 4, 256, 0.1).eval()
        with torch.no_grad():
            x, weights = layer(torch.randn(2, 7, 128))
        assert x.shape == (2, 7, 128)
        assert weights.shape == (2, 4, 7, 7)

    def test_post_norm_output(self):
        # The last operation is a LayerNorm with unit gain and zero bias
        layer = EncoderLayer(128, 4, 256, 0.1).eval()
        with torch.no_grad():
            x, _ = layer(torch.randn(2, 7, 128))
        assert torch.allclose(x.mean(dim=-1), torch.zeros(2, 7), atol=1e-5)

    def test_decoder_layer(self):
        layer = DecoderLayer(128, 4, 256, 0.1).eval()
        mask = causal_mask(4, torch.device('cpu'))
        with torch.no_grad():
            y, self_weights, cross_weights = layer(
                torch.randn(2, 4, 128), torch.randn(2, 7, 128), mask)
        assert y.shape == (2, 4, 128)
        assert self_weights.shape == (2, 4, 4, 4)
        assert cross_weights.shape == (2, 4, 4, 7)
        assert torch.all(self_weights[..., ~mask] == 0)


class TestModel:

    def test_forward_shape(self):
        model = Transformer().eval()
        src = torch.randint(0, 10, (3, 7))
        tgt_in = torch.randint(0, 10, (3, 4))
        with torch.no_grad():
            logits = model(src, tgt_in)
        assert logits.shape == (3, 4, VOCAB_SIZE)

    def test_attention_shapes(self):
        model = Transformer().eval()
        with torch.no_grad():
            weights = model.attention(torch.randint(0, 10, (3, 7)),
                                      torch.randint(0, 10, (3, 4)))
        expected = {'encoder_self': (3, 4, 7, 7),
                    'decoder_self': (3, 4, 4, 4),
                    'decoder_cross': (3, 4, 4, 7)}
        for key, shape in expected.items():
            assert [w.shape for w in weights[key]] == [shape] * 2

    def test_twenty_four_heads(self):
        modules = Transformer().attention_modules()
        assert len(modules) == 6
        assert sum(m.num_heads for _, m in modules) == 24

    def test_generate(self):
        model = Transformer().eval()
        predicted = generate(model, torch.randint(0, 10, (5, 7)))
        assert predicted.shape == (5, 4)
        assert predicted.dtype == torch.long
        assert predicted.min() >= 0 and predicted.max() < VOCAB_SIZE


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
