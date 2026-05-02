"""Distillation utilities.

Per the negative-result paper (arXiv 2509.01649), distillation hurts ICL after
~30% of pretraining tokens. We expose a `DistillScheduler` that anneals the KL
weight from `init` → `final` over a token budget and *hard-stops* at a cutoff.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def distillation_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 2.0,
) -> Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def sparse_topk_to_dense(
    topk_indices: Tensor,
    topk_logp: Tensor,
    vocab: int,
    fill: float = -1e9,
) -> Tensor:
    """Convert sparse top-k teacher logits (saved on disk) to dense logits.

    topk_indices: [..., K] long; topk_logp: [..., K] float.
    Returns dense logits [..., V] with `fill` outside the top-k.
    """
    *lead, K = topk_indices.shape
    out = topk_indices.new_full((*lead, vocab), 0).to(topk_logp.dtype) + fill
    out.scatter_(-1, topk_indices, topk_logp)
    return out


class DistillScheduler:
    """Anneals KL weight and hard-stops at a token cutoff."""

    def __init__(
        self,
        init_w: float = 0.7,
        final_w: float = 0.3,
        anneal_tokens: int = 250_000_000_000,
        cutoff_tokens: int = 300_000_000_000,
    ):
        self.init_w = init_w
        self.final_w = final_w
        self.anneal = anneal_tokens
        self.cutoff = cutoff_tokens

    def weight(self, tokens_seen: int) -> float:
        if tokens_seen >= self.cutoff:
            return 0.0
        if tokens_seen >= self.anneal:
            return self.final_w
        frac = tokens_seen / max(self.anneal, 1)
        return self.init_w + (self.final_w - self.init_w) * frac
