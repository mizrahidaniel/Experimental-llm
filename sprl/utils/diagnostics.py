"""Diagnostics for kill-criterion checks.

Implements the metrics referenced in Section 8.4 of the spec:
- Spearman ρ between log|Λ_t| and oracle next-token surprise (Bet A health).
- Power-law fit R² for loss-vs-iter (Bet C health).
- ‖T_b − I‖ (Bet C non-trivial).
- Tropical-head argmax position concentration (Bet B health).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation. Returns 0 if either input is constant."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return 0.0

    def rankdata(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1)
        # Average ties.
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        rank_avg = np.zeros_like(ranks)
        for i in range(len(_)):
            mask = inv == i
            rank_avg[mask] = ranks[mask].mean()
        return rank_avg

    rx = rankdata(x)
    ry = rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------


def power_law_fit(
    iters: Sequence[int], losses: Sequence[float]
) -> Tuple[float, float, float]:
    """Fit `loss = a * iter^(-b)` via log-log linear regression.

    Returns (a, b, r_squared). Used as a kill-criterion diagnostic for Bet C:
    if r_squared < 0.9 the RG-flow hypothesis fails.
    """
    iters = np.asarray(iters, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    mask = (iters > 0) & (losses > 0) & np.isfinite(losses)
    if mask.sum() < 3:
        return float("nan"), float("nan"), 0.0

    log_i = np.log(iters[mask])
    log_l = np.log(losses[mask])

    # If the response has zero variance, no power law can be fit.
    ss_tot = float(np.sum((log_l - log_l.mean()) ** 2))
    if ss_tot < 1.0e-12:
        return float("nan"), 0.0, 0.0

    slope, intercept = np.polyfit(log_i, log_l, 1)
    pred = slope * log_i + intercept
    ss_res = float(np.sum((log_l - pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    return float(math.exp(intercept)), float(-slope), float(r2)


# ---------------------------------------------------------------------------


def operator_distance_from_identity(T: torch.Tensor) -> float:
    """‖T − I‖_F. For Bet C, expect ≥ 0.1 — otherwise T_b collapsed to identity."""
    assert T.dim() == 2 and T.shape[0] == T.shape[1], "T must be square"
    eye = torch.eye(T.shape[0], device=T.device, dtype=T.dtype)
    return float(torch.linalg.norm(T - eye).item())


# ---------------------------------------------------------------------------


def argmax_concentration(soft_attn: torch.Tensor, hard_attn: torch.Tensor) -> float:
    """Fraction of positions where softmax-argmax == tropical-argmax.

    Computed over the last attention dim. >0.95 means tropical head is degenerate.
    """
    assert soft_attn.shape == hard_attn.shape
    soft_idx = soft_attn.argmax(dim=-1)
    hard_idx = hard_attn.argmax(dim=-1)
    return float((soft_idx == hard_idx).float().mean().item())


# ---------------------------------------------------------------------------


def log_det_lowrank(D: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """log det(D + U Uᵀ) via matrix determinant lemma.

    log det(D + U Uᵀ) = log det(D) + log det(I_r + Uᵀ D⁻¹ U)
    """
    assert D.dim() == U.dim() - 1
    # D: [..., d], U: [..., d, r]
    eps = 1.0e-8
    D_safe = D.clamp_min(eps)
    log_det_D = torch.log(D_safe).sum(dim=-1)
    DinvU = U / D_safe.unsqueeze(-1)  # [..., d, r]
    M = torch.einsum("...dr,...ds->...rs", U, DinvU)
    r = U.shape[-1]
    eye = torch.eye(r, device=U.device, dtype=U.dtype).expand_as(M)
    sign, logabsdet = torch.linalg.slogdet(eye + M)
    return log_det_D + logabsdet
