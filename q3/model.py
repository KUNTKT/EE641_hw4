"""
Decoder-only transformer language model with a pre-norm / post-norm
switch.

Masks are boolean, True where attention is allowed. Shapes use h for the
number of heads and d_k = d_model / h for the per-head dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Mask that lets position i attend to positions 0 through i.

    Returns:
        Tensor [size, size] (bool), True on and below the diagonal
    """
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()


class Block(nn.Module):
    """
    Decoder block, causal self-attention then FFN.

    'post' (Vaswani et al.):
        x = LayerNorm(x + Dropout(SelfAttention(x)))
        x = LayerNorm(x + Dropout(FFN(x)))
    'pre':
        x = x + Dropout(SelfAttention(LayerNorm(x)))
        x = x + Dropout(FFN(LayerNorm(x)))
    SelfAttention: q, k, v projections Linear(d_model -> d_model), split
        into h heads, F.scaled_dot_product_attention with the causal mask,
        heads concatenated, output projection Linear(d_model -> d_model)
    FFN: Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, norm: str = 'pre'):
        """
        Args:
            d_model: Model dimension
            num_heads: Attention heads h
            d_ff: FFN hidden dimension
            dropout: Dropout rate on sublayer outputs
            norm: LayerNorm placement, 'pre' or 'post'
        """
        super().__init__()
        assert norm in ('pre', 'post')
        self.num_heads = num_heads
        self.norm = norm
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def self_attention(self, x: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, T, d_model]
            mask: Causal mask [T, T] (bool), True where allowed

        Returns:
            Tensor [B, T, d_model]
        """
        B, T, d_model = x.shape

        def split(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        heads = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.W_o(heads.transpose(1, 2).reshape(B, T, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, T, d_model]
            mask: Causal mask [T, T] (bool), True where allowed

        Returns:
            Tensor [B, T, d_model]
        """
        if self.norm == 'pre':
            x = x + self.dropout(self.self_attention(self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self.self_attention(x, mask)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLM(nn.Module):
    """
    Token embedding plus learned position embedding, dropout, num_layers
    Blocks, a final LayerNorm (pre-norm only), and a linear head to
    vocabulary logits. The head is not tied to the token embedding.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 num_layers: int = 4, num_heads: int = 4, d_ff: int = 512,
                 max_len: int = 52, dropout: float = 0.1, norm: str = 'pre'):
        """
        Args:
            vocab_size: Number of token ids
            d_model: Model dimension
            num_layers: Number of Blocks
            num_heads: Attention heads per Block
            d_ff: FFN hidden dimension
            max_len: Number of learned positions
            dropout: Dropout rate on embeddings and sublayer outputs
            norm: LayerNorm placement in every Block, 'pre' or 'post'
        """
        super().__init__()
        assert norm in ('pre', 'post')
        self.norm = norm
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, d_ff, dropout, norm)
             for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model) if norm == 'pre' else None
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ids: Tensor [B, T] (long), T <= max_len

        Returns:
            logits: Tensor [B, T, vocab_size]
        """
        T = ids.shape[1]
        positions = torch.arange(T, device=ids.device)
        x = self.dropout(self.token_embedding(ids)
                         + self.position_embedding(positions))
        mask = causal_mask(T, ids.device)
        for block in self.blocks:
            x = block(x, mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return self.head(x)
