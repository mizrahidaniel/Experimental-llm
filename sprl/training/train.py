"""Tiny training-loop reference.

This is a *reference* loop: enough to drive the model end-to-end on synthetic
data so unit tests can exercise the full forward/backward, and so users on a
real 5080 box can substitute their data + sharding strategy without rewriting
the inner step.

It does NOT implement:
  - real distributed training (FSDP / TP) — single-device only.
  - real NVFP4 forward/backward — see `precision/nvfp4.py` for the policy.
  - real data loading from FineWeb-Edu — bring your own dataset iterator.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn

from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.training.losses import compute_total_loss
from sprl.training.optim import build_optimizer
from sprl.training.schedules import (
    alpha_ramp,
    beta_anneal,
    linear_warmup_cosine_decay,
)
from sprl.utils.logging import MetricLogger


def train_one_step(
    model: SPRLv2,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    cfg: SPRLConfig,
    step: int,
) -> dict:
    """Run one forward/backward/step on a single batch dict:
        patch_emb        [B, S, d_model]
        byte_targets     [B, S, max_patch_bytes]
        mtp_targets      [B, S, depth] (optional, if MTP enabled)
        teacher_logits   [B, S, max_patch_bytes, V] (optional)
    Returns metric dict.
    """
    optimizer.zero_grad(set_to_none=True)

    out = model.forward_from_patches(
        patch_emb=batch["patch_emb"],
        future_targets=batch.get("mtp_targets"),
    )

    # Beta annealing for tropical heads.
    if cfg.attention.tropical.enabled:
        warmup = max(int(cfg.training.total_steps * cfg.attention.tropical.warmup_fraction), 1)
        beta = beta_anneal(
            step, warmup,
            cfg.attention.tropical.beta_warmup_min,
            cfg.attention.tropical.beta_warmup_max,
        )
        for layer in model.layers:
            tropical = getattr(getattr(layer, "attn", None), "tropical", None)
            if tropical is not None:
                tropical.set_beta(beta)

    # Alpha ramp for RG-flow.
    rg_w = cfg.dram.rg_flow.alpha_target
    if cfg.dram.rg_flow.enabled:
        ramp = max(int(cfg.training.total_steps * cfg.dram.rg_flow.alpha_ramp_steps_frac), 1)
        rg_w = alpha_ramp(step, ramp, cfg.dram.rg_flow.alpha_target)

    loss, parts = compute_total_loss(
        out,
        byte_targets=batch["byte_targets"],
        teacher_logits=batch.get("teacher_logits"),
        mtp_targets=batch.get("mtp_targets"),
        mtp_weight=cfg.mtp.loss_weight if cfg.mtp.enabled else 0.0,
        rg_weight=rg_w,
        moe_aux_weight=1.0,
    )

    loss.backward()
    if cfg.training.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
    optimizer.step()

    # ALF bias-update (per step, post-optimizer).
    for layer in model.layers:
        ffn = getattr(layer, "ffn", None)
        if hasattr(ffn, "step_balancer"):
            ffn.step_balancer()

    # PI controller for κ.
    from sprl.recurrent.active_inference_router import ActiveInferenceRouter

    if isinstance(model.router, ActiveInferenceRouter):
        model.router.update_kappa(out.diagnostics["mean_k"])

    parts.update(out.diagnostics)
    return parts


def fit(
    model: SPRLv2,
    cfg: SPRLConfig,
    data_iter: Iterable[dict],
    logger: MetricLogger | None = None,
):
    optim = build_optimizer(
        model,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        optimizer_type=cfg.precision.optimizer_type,
        galore=cfg.precision.use_galore,
        galore_rank=cfg.precision.galore_rank,
        galore_targets=cfg.precision.galore_targets,
    )
    if logger is None:
        logger = MetricLogger(log_every=100)
    for step, batch in enumerate(data_iter):
        if step >= cfg.training.total_steps:
            break
        # LR schedule.
        lr = linear_warmup_cosine_decay(
            step, cfg.training.warmup_steps, cfg.training.total_steps, cfg.training.lr
        )
        for g in optim.param_groups:
            g["lr"] = lr
        metrics = train_one_step(model, batch, optim, cfg, step)
        metrics["lr"] = lr
        logger.log(metrics, step=step)
    return logger
