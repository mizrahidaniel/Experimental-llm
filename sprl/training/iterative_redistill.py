"""Iterative re-distillation pass (default off).

After the main distillation phase, run a held-out scoring pass; identify
documents where the student loss exceeds `loss_gap_threshold × teacher loss`;
re-train on the top fraction for `redistill_tokens` more.

This is the “hard-example mining” compose with distillation. Reference:
arXiv:2509.01649 caps standard distillation at ~30% of tokens; re-distillation
on the residual hard set is a way to push past that without hurting ICL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def loss_gap(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tuple[Tensor, Tensor]:
    """Per-document (student CE, teacher CE). Both [B,] from [B, S, V] inputs."""
    B, S, V = student_logits.shape
    s = F.cross_entropy(
        student_logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(B, S).mean(dim=-1)
    t = F.cross_entropy(
        teacher_logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(B, S).mean(dim=-1)
    return s, t


@dataclass
class HardExampleMiner:
    loss_gap_threshold: float = 1.2  # student/teacher ratio
    keep_fraction: float = 0.25      # top-k of held-out

    @torch.no_grad()
    def mine(
        self,
        batches: Iterable[Tuple[Tensor, Tensor, Tensor]],
        student,
    ) -> list[int]:
        """Walk a held-out iterator (yielding (logits_teacher, targets, doc_id))
        and return the doc_ids whose student loss exceeds the threshold ratio.
        """
        scored: list[tuple[float, int]] = []
        for teacher_logits, targets, doc_id in batches:
            student_logits = student(targets)  # caller's responsibility to wire
            s_loss, t_loss = loss_gap(student_logits, teacher_logits, targets)
            ratio = (s_loss / t_loss.clamp_min(1e-6)).item()
            if ratio >= self.loss_gap_threshold:
                scored.append((ratio, int(doc_id)))
        scored.sort(reverse=True)
        n_keep = max(1, int(len(scored) * self.keep_fraction))
        return [d for _, d in scored[:n_keep]]


def redistill_pass(
    model,
    optimizer,
    hard_doc_iter: Iterator,
    n_steps: int,
    train_step_fn,
) -> dict:
    """Drive a single re-distillation pass over the mined hard documents.

    `train_step_fn(model, batch, optimizer, step)` should return a metrics dict
    (the same callable used in the main loop).
    """
    history: list[dict] = []
    for s, batch in enumerate(hard_doc_iter):
        if s >= n_steps:
            break
        history.append(train_step_fn(model, batch, optimizer, s))
    if not history:
        return {"redistill_steps": 0}
    last = history[-1]
    return {
        "redistill_steps": len(history),
        "final_lm_loss": last.get("lm", 0.0),
        "mean_lm_loss": sum(h.get("lm", 0.0) for h in history) / len(history),
    }
