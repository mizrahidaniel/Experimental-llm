"""Byte-level encoder/decoder for BLT patches.

Encoder: variable-length byte chunk → single d_model patch embedding
   via a small transformer + cross-attention pool.

Decoder: single d_model patch embedding → variable-length byte logits
   via a small transformer with cross-attention onto the patch.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from sprl.patcher.blt_patcher import PatchPlan


class _SmallBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class ByteEncoder(nn.Module):
    """Encodes per-patch byte windows into one d_model embedding per patch.

    Operates batched-over-patches: pads patch bytes to `max_patch_bytes`.
    """

    def __init__(
        self,
        byte_dim: int = 384,
        patch_dim: int = 1024,
        n_layers: int = 4,
        max_patch_bytes: int = 16,
    ):
        super().__init__()
        self.max_patch_bytes = max_patch_bytes
        self.byte_dim = byte_dim
        self.patch_dim = patch_dim
        self.byte_embed = nn.Embedding(257, byte_dim, padding_idx=256)  # 256 = pad
        self.pos = nn.Embedding(max_patch_bytes, byte_dim)
        self.blocks = nn.ModuleList([_SmallBlock(byte_dim) for _ in range(n_layers)])
        self.query = nn.Parameter(torch.randn(byte_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(byte_dim, 4, batch_first=True)
        self.proj = nn.Linear(byte_dim, patch_dim)

    def forward(self, patch_bytes: Tensor, lengths: Tensor) -> Tensor:
        """Args:
          patch_bytes: [N_patches, max_patch_bytes] of bytes (256 = pad).
          lengths: [N_patches] true bytes per patch.

        Returns: [N_patches, patch_dim]
        """
        N, L = patch_bytes.shape
        pad_mask = torch.arange(L, device=patch_bytes.device).unsqueeze(0) >= lengths.unsqueeze(1)
        x = self.byte_embed(patch_bytes)
        pos = self.pos(torch.arange(L, device=patch_bytes.device))
        x = x + pos
        for block in self.blocks:
            x = block(x, mask=pad_mask)
        # Cross-attention pool with a single learned query.
        q = self.query.expand(N, 1, self.byte_dim)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=pad_mask, need_weights=False)
        pooled = pooled.squeeze(1)  # [N, byte_dim]
        return self.proj(pooled)


class ByteDecoder(nn.Module):
    """Decodes a single d_model patch embedding back to a sequence of byte logits.

    Tied input embedding is held externally (in ByteLMHead); we just produce
    pre-tied features that project against the byte vocabulary.
    """

    def __init__(
        self,
        patch_dim: int = 1024,
        byte_dim: int = 384,
        n_layers: int = 2,
        max_patch_bytes: int = 16,
    ):
        super().__init__()
        self.max_patch_bytes = max_patch_bytes
        self.byte_dim = byte_dim
        self.patch_to_byte = nn.Linear(patch_dim, byte_dim)
        self.pos = nn.Embedding(max_patch_bytes, byte_dim)
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(byte_dim, 4, batch_first=True) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.RMSNorm(byte_dim) for _ in range(n_layers)])
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(byte_dim, 4 * byte_dim),
                    nn.GELU(),
                    nn.Linear(4 * byte_dim, byte_dim),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, patch_emb: Tensor, n_bytes: int) -> Tensor:
        """Args:
          patch_emb: [N_patches, patch_dim]
          n_bytes: how many bytes to predict per patch.

        Returns: [N_patches, n_bytes, byte_dim]
        """
        N = patch_emb.shape[0]
        kv = self.patch_to_byte(patch_emb).unsqueeze(1)  # [N, 1, byte_dim]
        pos = self.pos(torch.arange(n_bytes, device=patch_emb.device))
        x = pos.unsqueeze(0).expand(N, -1, -1).clone()
        for ca, n, ff in zip(self.cross_attn, self.norms, self.ff):
            h = n(x)
            a, _ = ca(h, kv, kv, need_weights=False)
            x = x + a
            x = x + ff(n(x))
        return x
