"""Teacher-disagreement active data selection (Bet C — default off).

For each candidate batch, compute student↔teacher KL *without backprop* and
keep the top-1× by disagreement, with a diversity regularizer to prevent
collapse onto noisy/ambiguous text.

Wire-up:
  selector = DisagreementSelector(diversity_lambda=0.5, history=1024)
  for raw_batch in candidate_pool:           # 4× larger than desired
      kept = selector.select(student, raw_batch, teacher_logits, embeds, k=batch_size)

References:
  - ASK-LLM: arXiv:2402.09668
  - DSIR: arXiv:2302.03169
  - MiniLLM (reverse-KL distillation): arXiv:2306.08543
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def disagreement_score(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Per-document KL(student ‖ teacher) summed over the vocab axis.

    student_logits / teacher_logits: [B, S, V]. Returns [B] scores.
    Both must use the same vocabulary; see `sprl.training.distill` for the
    sparse top-K equivalent when teacher logits live on disk.
    """
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    # KL(t || s) per token, then mean over seq.
    kl_per_tok = (t * (t.clamp_min(1e-12).log() - s)).sum(dim=-1)
    return kl_per_tok.mean(dim=-1)


@dataclass
class DisagreementSelector:
    """Top-k by disagreement with cosine-similarity diversity regularizer.

    Score: `disagreement(doc) − λ · max_sim_to_recent(doc)`. λ tuned to keep
    selected-batch token entropy ≥ 0.8× random-batch entropy (a proxy for
    "no mode collapse"; see kill criterion C in spec §6.2).

    Embeddings (`doc_embeds`) are an external responsibility — typically the
    student's mean-pooled patch embeddings or a precomputed sentence-transformer.
    """

    diversity_lambda: float = 0.5
    history_size: int = 1024
    _history: List[Tensor] = field(default_factory=list)

    def _max_sim_to_history(self, embeds: Tensor) -> Tensor:
        if not self._history:
            return torch.zeros(embeds.shape[0], device=embeds.device, dtype=embeds.dtype)
        H = torch.stack(self._history, dim=0)  # [H, d]
        e_n = F.normalize(embeds, dim=-1)
        H_n = F.normalize(H, dim=-1)
        sim = e_n @ H_n.t()  # [B, H]
        return sim.max(dim=-1).values

    @torch.no_grad()
    def select(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        doc_embeds: Tensor,
        k: int,
    ) -> Tensor:
        """Returns indices of the k selected docs (out of doc_embeds.shape[0])."""
        disagreement = disagreement_score(student_logits, teacher_logits)
        diversity_pen = self._max_sim_to_history(doc_embeds)
        score = disagreement - self.diversity_lambda * diversity_pen

        k_eff = min(k, score.shape[0])
        top = score.topk(k_eff).indices

        # Update history with the selected embeddings.
        for i in top.tolist():
            self._history.append(doc_embeds[i].detach())
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size :]

        return top

    def reset(self) -> None:
        self._history.clear()


def batch_token_entropy(token_ids: Tensor, vocab_size: int) -> float:
    """Shannon entropy (in nats) of the marginal token distribution in a batch.

    Used as the diversity-collapse alarm: if this drops below 0.8× the
    random-batch entropy on the same pool, the diversity regularizer is too
    weak and the selector is collapsing to a narrow slice of the data.
    """
    flat = token_ids.flatten()
    if flat.numel() == 0:
        return 0.0
    counts = torch.bincount(flat, minlength=vocab_size).float()
    p = counts / counts.sum().clamp_min(1.0)
    # Skip zero-prob entries.
    p = p[p > 0]
    return float(-(p * p.log()).sum().item())
