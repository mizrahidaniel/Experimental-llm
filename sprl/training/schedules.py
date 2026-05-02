"""Learning-rate and regularizer schedules."""

from __future__ import annotations

import math


def linear_warmup_cosine_decay(
    step: int, warmup: int, total: int, base_lr: float, min_lr_frac: float = 0.1
) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


def alpha_ramp(step: int, ramp_steps: int, target: float) -> float:
    """Ramp from 0 → target over ramp_steps; constant afterwards."""
    if ramp_steps <= 0 or step >= ramp_steps:
        return target
    return target * (step / ramp_steps)


def beta_anneal(
    step: int, warmup_steps: int, beta_min: float, beta_max: float
) -> float:
    """Tropical-attention β schedule: beta_min → beta_max over warmup_steps."""
    if step >= warmup_steps:
        return beta_max
    frac = step / max(warmup_steps, 1)
    return beta_min + (beta_max - beta_min) * frac
