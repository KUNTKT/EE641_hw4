"""
Positional encodings: learned, sinusoidal, and rotary.

Conventions: positions count from 0. Frequencies are
theta_i = 10000^(-2i / d_model) for i = 0 .. d_model/2 - 1. Dimensions
pair as (2i, 2i+1), interleaved. The sinusoidal table holds
sin(pos * theta_i) at dimension 2i and cos(pos * theta_i) at 2i+1.
"""

import torch
import torch.nn as nn

BASE = 10000.0


class LearnedPositions(nn.Module):
    """Trainable position table, one d_model row per position."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(max_len, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Tensor [T] (long)

        Returns:
            Tensor [T, d_model]
        """
        return self.table(positions)


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """
    Sinusoidal position table (Vaswani et al. 2017).

    Args:
        max_len: Number of positions
        d_model: Embedding dimension (even)

    Returns:
        table: Tensor [max_len, d_model], row pos holds sin(pos * theta_i)
            at dimension 2i and cos(pos * theta_i) at dimension 2i+1
    """
    theta = rotary_frequencies(d_model)
    angles = torch.arange(max_len, dtype=torch.float32)[:, None] * theta
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table


def shift_rotation(shift: int, d_model: int) -> torch.Tensor:
    """
    The linear map that moves a sinusoidal table row forward by shift
    positions: table[pos + shift] = shift_rotation(shift, d_model) @
    table[pos] for every pos.

    Block-diagonal with one 2x2 block per pair (2i, 2i+1), from the
    angle-addition identities applied to the pair's (sin, cos) entries.

    Returns:
        rotation: Tensor [d_model, d_model]
    """
    # sin((p+s)t) =  cos(st) sin(pt) + sin(st) cos(pt)
    # cos((p+s)t) = -sin(st) sin(pt) + cos(st) cos(pt)
    angles = shift * rotary_frequencies(d_model)
    cos, sin = torch.cos(angles), torch.sin(angles)
    rotation = torch.zeros(d_model, d_model)
    even = torch.arange(0, d_model, 2)
    rotation[even, even] = cos
    rotation[even, even + 1] = sin
    rotation[even + 1, even] = -sin
    rotation[even + 1, even + 1] = cos
    return rotation


def rotary_frequencies(d_model: int) -> torch.Tensor:
    """
    Rotation frequency for each channel pair.

    Returns:
        theta: Tensor [d_model / 2], theta_i = 10000^(-2i / d_model)
    """
    exponents = torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
    return BASE ** (-exponents)


def apply_rotary(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Rotary position encoding (Su et al. 2021).

    Rotates each pair (2i, 2i+1) of the last dimension at position p by
    the angle p * theta_i:
        out[2i]   = x[2i] cos(p theta_i) - x[2i+1] sin(p theta_i)
        out[2i+1] = x[2i] sin(p theta_i) + x[2i+1] cos(p theta_i)

    Args:
        x: Tensor [..., T, d_model]
        positions: Tensor [T], position of each row of x

    Returns:
        Tensor [..., T, d_model]
    """
    theta = rotary_frequencies(x.shape[-1]).to(x.device)
    angles = positions.to(x.device, torch.float32)[:, None] * theta
    cos, sin = torch.cos(angles), torch.sin(angles)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def score_shift_invariance(q: torch.Tensor, k: torch.Tensor, shift: int,
                           scheme: str, query_pos: int = 12,
                           key_pos: int = 5) -> float:
    """
    Change in one attention score when the query and the key both move
    by shift positions.

    The score is the dot product of query and key after position
    encoding. 'rotary' rotates both with apply_rotary. 'sinusoidal' adds
    the table row of each position to each vector.

    Args:
        q: Query vector [d_model]
        k: Key vector [d_model]
        shift: Positions added to both query_pos and key_pos
        scheme: 'rotary' or 'sinusoidal'
        query_pos, key_pos: Unshifted positions

    Returns:
        |score(query_pos + shift, key_pos + shift)
            - score(query_pos, key_pos)|
    """
    assert scheme in ('rotary', 'sinusoidal')
    d_model = q.shape[-1]

    def score(qp: int, kp: int) -> float:
        if scheme == 'rotary':
            qe = apply_rotary(q[None], torch.tensor([qp]))[0]
            ke = apply_rotary(k[None], torch.tensor([kp]))[0]
        else:
            table = sinusoidal_table(max(qp, kp) + 1, d_model)
            qe, ke = q + table[qp], k + table[kp]
        return torch.dot(qe, ke).item()

    return abs(score(query_pos + shift, key_pos + shift)
               - score(query_pos, key_pos))
