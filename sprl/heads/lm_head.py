"""Plain LM head for the BPE path.

A simple `Linear(d_model, vocab_size)` projection with optional weight-tying
to the input embedding. At vocab=128K, d=768, weight-tying saves ~98M params.

For the BLT (byte-level) path see `sprl.heads.byte_lm_head.ByteLMHead`, which
expands per-patch latents back into byte sequences via cross-attention.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LMHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tied_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        if tied_embedding is not None:
            assert tied_embedding.embedding_dim == d_model
            assert tied_embedding.num_embeddings >= vocab_size
            self._tied = tied_embedding
            self.proj: Optional[nn.Linear] = None
        else:
            self._tied = None
            self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: Tensor) -> Tensor:
        """h: [..., d_model] → [..., vocab_size]."""
        if self._tied is not None:
            w = self._tied.weight[: self.vocab_size]
            return F.linear(h, w)
        return self.proj(h)
