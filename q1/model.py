"""
Encoder-decoder transformer for three-digit addition.

Layers are post-norm as in Vaswani et al.: each sublayer's output is
added to its input and the sum is layer-normalized. Every layer returns
its attention weights alongside its output, and Transformer.attention
collects them for analysis.
"""

import math

import torch
import torch.nn as nn

from attention import MultiHeadAttention, causal_mask
from dataset import BOS, SRC_LEN, TGT_LEN, VOCAB_SIZE


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    """
    Sinusoidal position table: PE[pos, 2i] = sin(pos / 10000^(2i/d_model)),
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model)).

    Returns:
        Tensor [max_len, d_model]
    """
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    inv_freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / d_model))
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(positions * inv_freq)
    table[:, 1::2] = torch.cos(positions * inv_freq)
    return table


class TokenEmbedding(nn.Module):
    """Token embedding with init std d_model^-1/2, scaled by sqrt(d_model)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=d_model ** -0.5)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor [B, T] (long)

        Returns:
            Tensor [B, T, d_model]
        """
        # Unit-scale entries after the multiply, the same order as the
        # sinusoidal positions, so neither signal drowns the other
        return self.embedding(tokens) * self.scale


class EncoderLayer(nn.Module):
    """
    Post-norm encoder layer:
        x = LayerNorm(x + Dropout(SelfAttention(x)))
        x = LayerNorm(x + Dropout(FFN(x)))
    FFN: Linear(d_model -> d_ff) -> ReLU -> Linear(d_ff -> d_model)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Tensor [B, T_src, d_model]

        Returns:
            x: Tensor [B, T_src, d_model]
            self_weights: Tensor [B, h, T_src, T_src]
        """
        attn, self_weights = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, self_weights


class DecoderLayer(nn.Module):
    """
    Post-norm decoder layer:
        y = LayerNorm(y + Dropout(MaskedSelfAttention(y)))
        y = LayerNorm(y + Dropout(CrossAttention(y, memory)))
        y = LayerNorm(y + Dropout(FFN(y)))
    Cross-attention takes queries from y and keys and values from the
    encoder output. FFN as in EncoderLayer.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, memory: torch.Tensor,
                mask: torch.Tensor) -> tuple:
        """
        Args:
            y: Tensor [B, T_tgt, d_model]
            memory: Encoder output [B, T_src, d_model]
            mask: Causal mask [T_tgt, T_tgt] (bool), True where allowed

        Returns:
            y: Tensor [B, T_tgt, d_model]
            self_weights: Tensor [B, h, T_tgt, T_tgt]
            cross_weights: Tensor [B, h, T_tgt, T_src]
        """
        attn, self_weights = self.self_attn(y, y, y, mask)
        y = self.norm1(y + self.dropout(attn))
        attn, cross_weights = self.cross_attn(y, memory, memory)
        y = self.norm2(y + self.dropout(attn))
        y = self.norm3(y + self.dropout(self.ffn(y)))
        return y, self_weights, cross_weights


class Transformer(nn.Module):
    """
    Encoder-decoder stack. Source and target share one token embedding
    and one sinusoidal position table. Dropout follows the embedding sum.
    A linear head maps the last decoder layer to vocabulary logits.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 128,
                 num_heads: int = 4, d_ff: int = 256, num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            vocab_size: Number of token ids
            d_model: Model dimension
            num_heads: Heads per attention module
            d_ff: FFN hidden dimension
            num_layers: Layers in each of the encoder and decoder
            dropout: Dropout rate on embeddings and sublayer outputs
        """
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.register_buffer('positions', sinusoidal_positions(
            max(SRC_LEN, TGT_LEN), d_model))
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.output = nn.Linear(d_model, vocab_size)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embedding plus position, [B, T] -> [B, T, d_model]."""
        positions = self.positions[:tokens.shape[1]]
        return self.dropout(self.embed(tokens) + positions)

    def encode(self, src: torch.Tensor) -> tuple:
        """
        Args:
            src: Tensor [B, T_src] (long)

        Returns:
            memory: Tensor [B, T_src, d_model]
            self_weights: List over layers of [B, h, T_src, T_src]
        """
        x = self.embed_tokens(src)
        self_weights = []
        for layer in self.encoder:
            x, weights = layer(x)
            self_weights.append(weights)
        return x, self_weights

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor) -> tuple:
        """
        Args:
            tgt_in: Tensor [B, T_tgt] (long), starting with BOS
            memory: Encoder output [B, T_src, d_model]

        Returns:
            logits: Tensor [B, T_tgt, vocab_size]
            self_weights: List over layers of [B, h, T_tgt, T_tgt]
            cross_weights: List over layers of [B, h, T_tgt, T_src]
        """
        y = self.embed_tokens(tgt_in)
        mask = causal_mask(tgt_in.shape[1], tgt_in.device)
        self_weights, cross_weights = [], []
        for layer in self.decoder:
            y, s_weights, c_weights = layer(y, memory, mask)
            self_weights.append(s_weights)
            cross_weights.append(c_weights)
        return self.output(y), self_weights, cross_weights

    def forward(self, src: torch.Tensor,
                tgt_in: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            src: Tensor [B, T_src] (long)
            tgt_in: Tensor [B, T_tgt] (long), starting with BOS

        Returns:
            logits: Tensor [B, T_tgt, vocab_size]
        """
        memory, _ = self.encode(src)
        logits, _, _ = self.decode(tgt_in, memory)
        return logits

    def attention(self, src: torch.Tensor, tgt_in: torch.Tensor) -> dict:
        """
        Every attention module's weights for one forward pass.

        Returns:
            Dict with 'encoder_self', 'decoder_self', 'decoder_cross',
            each a list over layers of [B, h, T_q, T_k]
        """
        memory, encoder_self = self.encode(src)
        _, decoder_self, decoder_cross = self.decode(tgt_in, memory)
        return {'encoder_self': encoder_self,
                'decoder_self': decoder_self,
                'decoder_cross': decoder_cross}

    def attention_modules(self) -> list:
        """
        Every MultiHeadAttention module with a name, in the order
        enc0_self, enc1_self, dec0_self, dec0_cross, dec1_self,
        dec1_cross.

        Returns:
            List of (name, MultiHeadAttention)
        """
        modules = [(f'enc{i}_self', layer.self_attn)
                   for i, layer in enumerate(self.encoder)]
        for i, layer in enumerate(self.decoder):
            modules.append((f'dec{i}_self', layer.self_attn))
            modules.append((f'dec{i}_cross', layer.cross_attn))
        return modules


@torch.no_grad()
def generate(model, src: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding: start from BOS and append the argmax token four
    times. Call with the model in eval mode.

    Args:
        src: Tensor [B, T_src] (long)

    Returns:
        Predicted target digits [B, 4] (long), in the model's digit order
    """
    memory, _ = model.encode(src)
    tokens = torch.full((src.shape[0], 1), BOS, dtype=torch.long,
                        device=src.device)
    for _ in range(TGT_LEN):
        logits, _, _ = model.decode(tokens, memory)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens[:, 1:]
