"""Byte-level LM head with cross-attention from per-patch latent → bytes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sprl.patcher.byte_encoder import ByteDecoder


class ByteLMHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        byte_dim: int = 384,
        n_layers: int = 2,
        max_patch_bytes: int = 16,
        vocab: int = 256,
        tie_weights_with: nn.Embedding | None = None,
    ):
        super().__init__()
        self.decoder = ByteDecoder(
            patch_dim=d_model,
            byte_dim=byte_dim,
            n_layers=n_layers,
            max_patch_bytes=max_patch_bytes,
        )
        self.norm = nn.RMSNorm(byte_dim)
        if tie_weights_with is not None:
            assert tie_weights_with.embedding_dim == byte_dim
            assert tie_weights_with.num_embeddings >= vocab
            self.head = lambda h: F.linear(h, tie_weights_with.weight[:vocab])
            self._head_param = tie_weights_with
        else:
            self.head = nn.Linear(byte_dim, vocab, bias=False)
            self._head_param = None

    def forward(self, patch_emb: Tensor, n_bytes: int) -> Tensor:
        """patch_emb: [N_patches, d_model]. Returns logits [N_patches, n_bytes, vocab]."""
        h = self.decoder(patch_emb, n_bytes)
        h = self.norm(h)
        return self.head(h)
