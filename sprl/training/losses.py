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
    distill_kl_direction: str = "teacher_forward_kl",
    mtp_targets: Optional[Tensor] = None,
    mtp_weight: float = 0.3,
    rg_weight: float = 1.0e-3,
    moe_aux_weight: float = 1.0,
    compute_budget_weight: float = 1.0,
) -> tuple[Tensor, Dict[str, float]]:
    """Returns (total_loss, breakdown).

    byte_targets: int targets aligned with `out.byte_logits`. Shape:
      - BLT path: [B, S_patches, max_patch_bytes]
      - BPE path: [B, S_tokens]
      Use -100 for ignore.
    teacher_logits: soft KD targets, same shape as `out.byte_logits` but in
      the *distillation* vocab. For teacher_token_kl that's the student vocab;
      for auxiliary_teacher_token_head that's the teacher vocab and is paired
      with `out.aux_logits` (the LM head still trains on student-vocab CE).
    """
    parts: Dict[str, float] = {}

    # Main LM loss. Reshape-to-flat handles both BLT [B, S, Bp, V] and
    # BPE [B, S, V] shapes.
    logits = out.byte_logits
    V = logits.shape[-1]
    lm_loss = F.cross_entropy(
        logits.reshape(-1, V),
        byte_targets.reshape(-1),
        ignore_index=-100,
    )
    parts["lm"] = float(lm_loss.item())

    total = lm_loss

    # Distillation.
    if teacher_logits is not None and distill_kl_weight > 0.0:
        # When the model has an auxiliary teacher-token head (vocab mismatch
        # path), distill against `aux_logits` instead of the main LM head's
        # logits — those are in the wrong vocab space.
        kl_logits = out.aux_logits if out.aux_logits is not None else logits
        if kl_logits.shape[-1] != teacher_logits.shape[-1]:
            raise ValueError(
                "teacher_logits vocab does not match the distillation head "
                f"({kl_logits.shape[-1]} vs {teacher_logits.shape[-1]}). "
                "If your tokenizers don't align, use "
                "distill_mode=auxiliary_teacher_token_head."
            )
        T = 2.0
        Vd = kl_logits.shape[-1]
        if distill_kl_direction == "teacher_forward_kl":
            student_logp = F.log_softmax(kl_logits.reshape(-1, Vd) / T, dim=-1)
            teacher_p = F.softmax(teacher_logits.reshape(-1, Vd) / T, dim=-1)
            kl = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (T ** 2)
        elif distill_kl_direction == "reverse_kl":
            student_p = F.softmax(kl_logits.reshape(-1, Vd) / T, dim=-1)
            teacher_logp = F.log_softmax(teacher_logits.reshape(-1, Vd) / T, dim=-1)
            kl = F.kl_div(teacher_logp, student_p, reduction="batchmean") * (T ** 2)
        else:
            raise ValueError(f"unknown distill_kl_direction: {distill_kl_direction!r}")
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
