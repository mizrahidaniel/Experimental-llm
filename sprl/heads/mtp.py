"""Multi-Token Prediction (MTP) head.

DeepSeek-V3-style speculative decoding: predicts the next `depth` tokens via a
shallow auxiliary stack. Loss is added at training (weight 0.3 by default);
at inference it enables ≥2× speculative-decoding speedup.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _MTPLayer(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.emb_proj = nn.Linear(d_model, d_model, bias=False)
        self.h_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_out = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, h_prev: Tensor, target_tok_emb: Tensor) -> tuple[Tensor, Tensor]:
        """h_prev: [N, d]; target_tok_emb: [N, d]. Returns (h_next, logits)."""
        h = self.norm(h_prev)
        x = self.h_proj(h) + self.emb_proj(target_tok_emb)
        x = x + self.ffn(self.norm_out(x))
        logits = self.head(x)
        return x, logits


class MultiTokenPredictionHead(nn.Module):
    def __init__(self, d_model: int, depth: int = 2, vocab: int = 256):
        super().__init__()
        self.depth = depth
        self.vocab = vocab
        self.layers = nn.ModuleList([_MTPLayer(d_model, vocab) for _ in range(depth)])
        self.embed_for_targets = nn.Embedding(vocab, d_model)

    def forward(self, h: Tensor, future_targets: Tensor) -> list[Tensor]:
        """h: [B, S, d]. future_targets: [B, S, depth] (next 2 tokens per step).

        Returns a list of [B, S, vocab] logit tensors, one per future-step.
        """
        h_flat = h.reshape(-1, h.shape[-1])
        emb = self.embed_for_targets(future_targets)  # [B, S, depth, d]
        emb_flat = emb.reshape(-1, self.depth, h.shape[-1])

        outputs = []
        h_cur = h_flat
        for d in range(self.depth):
            h_cur, logits = self.layers[d](h_cur, emb_flat[:, d])
            outputs.append(logits.reshape(*h.shape[:-1], self.vocab))
        return outputs

    @staticmethod
    def loss(logits_list: list[Tensor], targets: Tensor, weight: float = 0.3) -> Tensor:
        """logits_list: list of [B, S, vocab]; targets: [B, S, depth]."""
        losses = []
        for d, logits in enumerate(logits_list):
            tgt = targets[..., d]
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                ignore_index=-100,
            ))
        return weight * torch.stack(losses).mean()
