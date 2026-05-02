"""Sliding-window causal attention.

Used on odd-numbered layers in the alternating ASA pattern (Section 4.3.3).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sprl.attention.rope import build_rope_cache, apply_rope


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 1024,
        rope_base: float = 500_000.0,
        causal: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.causal = causal
        self.rope_base = rope_base

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, h: Tensor) -> Tensor:
        B, S, _ = h.shape
        Q = self.Wq(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(
            S, self.head_dim, base=self.rope_base, device=h.device, dtype=h.dtype
        )
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Build sliding-window + causal mask.
        i = torch.arange(S, device=h.device)
        rel = i.unsqueeze(0) - i.unsqueeze(1)  # [S, S]; positive = future
        if self.causal:
            allowed = (rel <= 0) & (rel > -self.window_size)
        else:
            allowed = rel.abs() < self.window_size
        scores = scores.masked_fill(~allowed, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, S, self.d_model)
        return self.Wo(out)
