"""Tiny byte-level LM that supplies entropy estimates for the BLT patcher.

In production, this is a fixed (frozen) ~10M-param oracle pretrained on 30B
bytes. Here we provide a small reference model (~few hundred K params) so unit
tests can run anywhere; you're expected to replace the weights with a real
pretrained byte-LM checkpoint before using BLT in earnest.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class ByteLM(nn.Module):
    """Tiny causal byte LM. Standard pre-norm transformer."""

    def __init__(self, dim: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(256, dim)
        self.layers = nn.ModuleList(
            [_ByteLMBlock(dim, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, 256, bias=False)

    def forward(self, bytes_in: Tensor) -> Tensor:
        """bytes_in: [batch, T] of int in [0,255]. Returns logits [batch, T, 256]."""
        x = self.embed(bytes_in)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def next_byte_entropy(self, bytes_in: Tensor) -> Tensor:
        """Return per-position entropy (nats) of the predicted next-byte distribution.

        Output shape [batch, T] aligned with `bytes_in`. Entropy at position t is
        H(p(b_{t+1} | b_{≤t})) — i.e. predicted entropy *before observing* b_{t+1}.
        """
        logits = self.forward(bytes_in)
        log_p = torch.log_softmax(logits, dim=-1)
        p = log_p.exp()
        H = -(p * log_p).sum(dim=-1)
        return H


class _ByteLMBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(dim)
        hidden = int(dim * 4)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[1]
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x
