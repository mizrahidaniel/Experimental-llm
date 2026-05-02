"""Total-loss assembly.

Mirrors the equation set in Section 7 of the spec.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from sprl.model import SPRLOutput


def compute_total_loss(
    out: SPRLOutput,
    byte_targets: Tensor,
    teacher_logits: Optional[Tensor] = None,
    *,
    distill_kl_weight: float = 0.0,
    mtp_targets: Optional[Tensor] = None,
    mtp_weight: float = 0.3,
    rg_weight: float = 1.0e-3,
    moe_aux_weight: float = 1.0,
    compute_budget_weight: float = 1.0,
) -> tuple[Tensor, Dict[str, float]]:
    """Returns (total_loss, breakdown).

    byte_targets: [B, S_patches, n_bytes] — int. Use -100 for ignore.
    teacher_logits: [B, S_patches, n_bytes, vocab] — soft KD targets (log-probs OK,
        we'll normalize). Pass None if no distillation.
    mtp_targets: [B, S_patches, depth] — for MTP head.
    """
    parts: Dict[str, float] = {}

    # Main LM loss.
    logits = out.byte_logits  # [B, S, B_p, V]
    B, S, Bp, V = logits.shape
    lm_loss = F.cross_entropy(
        logits.reshape(-1, V),
        byte_targets.reshape(-1),
        ignore_index=-100,
    )
    parts["lm"] = float(lm_loss.item())

    total = lm_loss

    # Distillation.
    if teacher_logits is not None and distill_kl_weight > 0.0:
        T = 2.0
        student_logp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
        teacher_p = F.softmax(teacher_logits.reshape(-1, V) / T, dim=-1)
        kl = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (T ** 2)
        parts["kl_distill"] = float(kl.item())
        total = (1 - distill_kl_weight) * total + distill_kl_weight * kl

    # MTP loss.
    if out.mtp_logits is not None and mtp_targets is not None:
        from sprl.heads.mtp import MultiTokenPredictionHead

        mtp_loss = MultiTokenPredictionHead.loss(
            out.mtp_logits, mtp_targets, weight=mtp_weight
        )
        parts["mtp"] = float(mtp_loss.item())
        total = total + mtp_loss

    # RG-flow regularizer.
    if "rg_flow" in out.aux_losses:
        rg = out.aux_losses["rg_flow"]
        parts["rg"] = float(rg.item())
        total = total + rg_weight * rg

    # MoE aux balance.
    if "moe_aux" in out.aux_losses:
        ax = out.aux_losses["moe_aux"]
        parts["moe_aux"] = float(ax.item())
        total = total + moe_aux_weight * ax

    # Compute-budget Lagrangian.
    if "compute_budget" in out.aux_losses:
        cb = out.aux_losses["compute_budget"]
        parts["compute_budget"] = float(cb.item())
        total = total + compute_budget_weight * cb

    parts["total"] = float(total.item())
    return total, parts
