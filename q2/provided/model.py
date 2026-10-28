"""
One-layer attention model for offset retrieval. Provided — do not modify.

Imports sinusoidal_table and apply_rotary from positional.py. Until
those are implemented, the 'sinusoidal' model raises NotImplementedError
at construction and the 'rotary' model at its forward pass.
"""

import math

import torch
import torch.nn as nn

from positional import LearnedPositions, apply_rotary, sinusoidal_table

POSITIONS = ('none', 'learned', 'sinusoidal', 'rotary')


class OffsetModel(nn.Module):
    """
    Token embedding, one causal single-head self-attention layer, and a
    linear output head. No feed-forward sublayer.

    h = embedding(tokens) + position rows ('learned', 'sinusoidal')
    q, k, v = W_q h, W_k h, W_v h, then 'rotary' rotates q and k
    weights = softmax(q k^T / sqrt(d_model)), key j > query i masked
    logits = head(LayerNorm(h + W_o (weights v)))
    """

    def __init__(self, vocab_size: int = 16, d_model: int = 64,
                 max_len: int = 128, position: str = 'rotary'):
        """
        Args:
            vocab_size: Number of token values
            d_model: Embedding and attention dimension
            max_len: Longest sequence. The 'learned' table has this many
                rows whether or not training reaches them
            position: 'none', 'learned', 'sinusoidal', or 'rotary'
        """
        super().__init__()
        assert position in POSITIONS
        self.position = position
        self.d_model = d_model
        # No sqrt(d_model) multiply: on N(0, 1) embeddings it would make
        # the unit-amplitude position signal 1/8 of the token signal.
        self.embedding = nn.Embedding(vocab_size, d_model)
        if position == 'learned':
            self.positions = LearnedPositions(max_len, d_model)
        elif position == 'sinusoidal':
            self.register_buffer('table', sinusoidal_table(max_len, d_model),
                                 persistent=False)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> tuple:
        """
        Args:
            tokens: Tensor [B, T] (long), T <= max_len

        Returns:
            logits: Tensor [B, T, vocab_size]
            weights: Attention weights [B, T, T], row i over keys 0..i
        """
        T = tokens.shape[1]
        positions = torch.arange(T, device=tokens.device)
        h = self.embedding(tokens)
        if self.position == 'learned':
            h = h + self.positions(positions)
        elif self.position == 'sinusoidal':
            h = h + self.table[:T]

        q, k, v = self.W_q(h), self.W_k(h), self.W_v(h)
        if self.position == 'rotary':
            q = apply_rotary(q, positions)
            k = apply_rotary(k, positions)
        scores = q @ k.transpose(1, 2) / math.sqrt(self.d_model)
        causal = torch.ones(T, T, dtype=torch.bool,
                            device=tokens.device).tril()
        weights = scores.masked_fill(~causal, float('-inf')).softmax(dim=-1)

        h = self.norm(h + self.W_o(weights @ v))
        return self.head(h), weights
