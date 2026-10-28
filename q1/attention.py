"""
Scaled dot-product attention, multi-head attention, and the causal mask.

Masks are boolean, True where attention is allowed. Shapes use h for the
number of heads and d_k = d_model / h for the per-head dimension.
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: torch.Tensor = None) -> tuple:
    """
    softmax(q k^T / sqrt(d_k)) v, with disallowed positions removed
    before the softmax.

    Args:
        q: Queries [B, h, T_q, d_k]
        k: Keys [B, h, T_k, d_k]
        v: Values [B, h, T_k, d_k]
        mask: Tensor broadcastable to [B, h, T_q, T_k] (bool), True
            where attention is allowed. None allows every position

    Returns:
        output: Tensor [B, h, T_q, d_k]
        weights: Tensor [B, h, T_q, T_k], each row sums to 1
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Mask that lets position i attend to positions 0 through i.

    Returns:
        Tensor [size, size] (bool), True on and below the diagonal
    """
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with learned projections W_q, W_k, W_v, W_o,
    each nn.Linear(d_model, d_model).

    Head i reads columns i*d_k to (i+1)*d_k of the projected queries,
    keys, and values. The concatenated head outputs pass through W_o, so
    head i's output enters W_o through input columns i*d_k to
    (i+1)*d_k. Zeroing those columns of W_o.weight removes head i.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model: Model dimension, divisible by num_heads
            num_heads: Number of heads h
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, h, T, d_k], head i = columns i*d_k..."""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Args:
            query: Tensor [B, T_q, d_model]
            key: Tensor [B, T_k, d_model]
            value: Tensor [B, T_k, d_model]
            mask: Tensor broadcastable to [B, h, T_q, T_k] (bool),
                True where attention is allowed. None allows every
                position

        Returns:
            output: Tensor [B, T_q, d_model]
            weights: Tensor [B, h, T_q, T_k]
        """
        q = self._split(self.W_q(query))
        k = self._split(self.W_k(key))
        v = self._split(self.W_v(value))
        heads, weights = scaled_dot_product_attention(q, k, v, mask)
        B, _, T_q, _ = heads.shape
        concat = heads.transpose(1, 2).reshape(B, T_q, self.d_model)
        return self.W_o(concat), weights
