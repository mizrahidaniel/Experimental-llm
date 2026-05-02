"""Distillation utilities.

Per the negative-result paper (arXiv 2509.01649), distillation hurts ICL after
~30% of pretraining tokens. We expose a `DistillScheduler` that anneals the KL
weight from `init` -> `final` over a token budget and *hard-stops* at a cutoff.

The offline-teacher path (top-K sparse logits stored on disk) lives in
`sprl.training.teacher_logits`; this module adds a combined-loss helper that
mixes hard CE with sparse-KL via the schedule weight beta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from sprl.training.teacher_logits import (
    TeacherLogitDataset,
    TeacherPack,
    batch_mixer,
    sparse_top_k_kl_loss,
)


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


@dataclass
class DistillScheduler:
    """Anneals KL weight and hard-stops at a token cutoff.

    `cutoff_tokens=0` disables distillation entirely (always returns 0.0).
    """

    init_w: float = 0.7
    final_w: float = 0.3
    anneal_tokens: int = 250_000_000_000
    cutoff_tokens: int = 300_000_000_000

    def weight(self, tokens_seen: int) -> float:
        if self.cutoff_tokens <= 0:
            return 0.0
        if tokens_seen >= self.cutoff_tokens:
            return 0.0
        if tokens_seen >= self.anneal_tokens:
            return self.final_w
        frac = tokens_seen / max(self.anneal_tokens, 1)
        return self.init_w + (self.final_w - self.init_w) * frac

    # Back-compat: older code referenced `.anneal` and `.cutoff` directly.
    @property
    def anneal(self) -> int:
        return self.anneal_tokens

    @property
    def cutoff(self) -> int:
        return self.cutoff_tokens


def combined_loss(
    student_logits: Tensor,
    byte_targets: Tensor,
    teacher_pack: Optional[TeacherPack],
    beta: float,
    T: float = 2.0,
    ignore_index: int = -100,
) -> tuple[Tensor, dict[str, float]]:
    """Hard CE + sparse top-K KL combined per arXiv 2509.01649.

    `loss = (1 - beta) * CE + beta * KL_sparse(teacher || student, T)`.

    `student_logits` shape `[..., V]` and `byte_targets` shape `[...]` (any
    leading dims). `teacher_pack.topk_*` shapes must match leading dims of
    `student_logits`.
    """
    V = student_logits.shape[-1]
    s_flat = student_logits.reshape(-1, V)
    t_flat = byte_targets.reshape(-1)
    ce = F.cross_entropy(s_flat, t_flat, ignore_index=ignore_index)
    parts = {"ce": float(ce.item())}

    if teacher_pack is None or beta <= 0.0:
        parts["kl"] = 0.0
        parts["total"] = float(ce.item())
        return ce, parts

    kl = sparse_top_k_kl_loss(
        student_logits,
        teacher_pack.topk_indices,
        teacher_pack.topk_logits,
        T=T,
    )
    parts["kl"] = float(kl.item())
    total = (1.0 - beta) * ce + beta * kl
    parts["total"] = float(total.item())
    return total, parts


def teacher_batches(
    root: str,
    batch_size: int,
    prefetch: int = 1,
    start_chunk_id: int = 0,
    start_token_offset: Optional[int] = None,
) -> Iterator[tuple[Tensor, TeacherPack]]:
    """Convenience wrapper: stream `(byte_targets, TeacherPack)` from a root."""
    ds = TeacherLogitDataset(
        root,
        prefetch=prefetch,
        start_chunk_id=start_chunk_id,
        start_token_offset=start_token_offset,
    )
    yield from batch_mixer(ds, batch_size)
