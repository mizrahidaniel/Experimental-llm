"""Rotary position embedding utilities.

Standard RoPE used by the decoupled rotary key in MLA.
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_rope_cache(
    seqlen: int, head_dim: int, base: float = 500_000.0, device=None, dtype=torch.float32
) -> tuple[Tensor, Tensor]:
    """Returns (cos, sin) of shape [seqlen, head_dim/2]."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seqlen, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [seqlen, head_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to the last dim of x.

    x: [..., seqlen, head_dim]
    cos, sin: [seqlen, head_dim/2]
    """
    seqlen = x.shape[-2]
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # Broadcast cos/sin over the leading dims.
    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out
