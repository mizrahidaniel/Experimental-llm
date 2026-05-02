"""Depth-attention.

At iteration k, attend over previous iterations {z_0, …, z_{k-1}} at the same
token position. Keeps the model honest about what it has produced so far,
following DRAM (arXiv 2601.21582).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DepthAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, z_k: Tensor, z_history: list[Tensor]) -> Tensor:
        """z_k: [B, S, d_model]. z_history: list of [B, S, d_model] for iters 0..k-1.

        Attention is along the iteration axis (per-token). When history is
        empty (k = 0), returns zeros.
        """
        if len(z_history) == 0:
            return torch.zeros_like(z_k)

        # Stack history as a new "iter" dim: [k, B, S, d]
        H = torch.stack(z_history, dim=0)  # [k, B, S, d]
        k = H.shape[0]
        B, S, d = z_k.shape

        # Q from z_k, K/V from H. Move to per-position attention along iter dim.
        # Shape Q: [B, S, n_h, hd] → [B, S, n_h, 1, hd]
        Q = self.Wq(z_k).view(B, S, self.n_heads, self.head_dim).unsqueeze(-2)
        K = self.Wk(H.permute(1, 2, 0, 3)).view(B, S, k, self.n_heads, self.head_dim)
        V = self.Wv(H.permute(1, 2, 0, 3)).view(B, S, k, self.n_heads, self.head_dim)

        # Move heads next to the seq dim: [B, S, n_h, k, hd]
        K = K.permute(0, 1, 3, 2, 4)
        V = V.permute(0, 1, 3, 2, 4)

        scores = (Q * K).sum(dim=-1) * self.scale  # [B, S, n_h, k]
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)
        out = (attn * V).sum(dim=-2)  # [B, S, n_h, hd]
        out = out.reshape(B, S, self.d_model)
        return self.Wo(out)
