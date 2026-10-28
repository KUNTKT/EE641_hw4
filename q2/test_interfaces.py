"""
Interface tests for Problem 2.

These verify shapes, the specified position conventions, and the two
encoding checks against hand-computed cases. They do not require
training and do not test learned behavior.

Run from q2/:  pytest test_interfaces.py
"""

import os

import pytest
import torch

from dataset import IGNORE_INDEX, OffsetDataset
from positional import (apply_rotary, rotary_frequencies,
                        score_shift_invariance, shift_rotation,
                        sinusoidal_table)
from provided.model import POSITIONS, OffsetModel

DATA_ROOT = '../datasets/offset'


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT),
                    reason='run generate_datasets.py first')
class TestDataset:

    def test_item_contract(self):
        dataset = OffsetDataset(DATA_ROOT, 'train', k=7)
        tokens, targets = dataset[0]
        assert tokens.shape == targets.shape == (64,)
        assert tokens.dtype == targets.dtype == torch.long
        assert (targets[:7] == IGNORE_INDEX).all()
        assert torch.equal(targets[7:], tokens[:-7])

    def test_test_split_length(self):
        tokens, _ = OffsetDataset(DATA_ROOT, 'test', k=7)[0]
        assert tokens.shape == (128,)


class TestSinusoidal:

    def test_table_shape_and_position_zero(self):
        table = sinusoidal_table(128, 64)
        assert table.shape == (128, 64)
        # sin(0) = 0 at even dimensions, cos(0) = 1 at odd
        assert torch.allclose(table[0, 0::2], torch.zeros(32))
        assert torch.allclose(table[0, 1::2], torch.ones(32))

    def test_first_pair(self):
        # theta_0 = 1, so pair 0 at position 3 is (sin 3, cos 3)
        table = sinusoidal_table(8, 4)
        assert torch.allclose(table[3, :2],
                              torch.tensor([0.14112, -0.98999]), atol=1e-4)

    def test_shift_is_rotation(self):
        table = sinusoidal_table(128, 64)
        for pos, shift in [(5, 3), (37, 20)]:
            rotation = shift_rotation(shift, 64)
            assert rotation.shape == (64, 64)
            error = (table[pos + shift] - rotation @ table[pos]).abs().max()
            assert error < 1e-4


class TestRotary:

    def test_frequencies(self):
        theta = rotary_frequencies(64)
        assert theta.shape == (32,)
        assert theta[0] == 1.0
        assert (theta[1:] < theta[:-1]).all()

    def test_identity_at_position_zero(self):
        x = torch.randn(1, 64)
        assert torch.allclose(apply_rotary(x, torch.tensor([0])), x)

    def test_preserves_pair_norms(self):
        x = torch.randn(2, 10, 64)
        rotated = apply_rotary(x, torch.arange(10))
        assert rotated.shape == x.shape
        norms = x.view(2, 10, 32, 2).norm(dim=-1)
        assert torch.allclose(rotated.view(2, 10, 32, 2).norm(dim=-1), norms,
                              atol=1e-5)

    def test_interleaved_pairs(self):
        # Pair 0 (dimensions 0, 1) rotates by position * 1 radian
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        rotated = apply_rotary(x, torch.tensor([1]))
        assert torch.allclose(rotated[0, :2], torch.tensor([0.54030, 0.84147]),
                              atol=1e-4)


class TestShiftInvariance:

    def test_rotary_invariant(self):
        q, k = torch.randn(64), torch.randn(64)
        assert score_shift_invariance(q, k, 20, 'rotary') < 1e-4

    def test_sinusoidal_not_invariant(self):
        torch.manual_seed(641)
        q, k = torch.randn(64), torch.randn(64)
        assert score_shift_invariance(q, k, 20, 'sinusoidal') > 1e-2


class TestModel:

    def test_output_shapes(self):
        for position in POSITIONS:
            model = OffsetModel(16, 64, 128, position).eval()
            logits, weights = model(torch.randint(0, 16, (3, 40)))
            assert logits.shape == (3, 40, 16)
            assert weights.shape == (3, 40, 40)
            assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 40))
            assert (weights.triu(diagonal=1) == 0).all()

    def test_learned_table_rows(self):
        model = OffsetModel(16, 64, 128, 'learned')
        assert model.positions.table.weight.shape == (128, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
