"""Streaming top-r SVD utilities (Brand 2002, "Fast low-rank modifications of
the thin SVD").

Used by the strict-rank info-form Kalman combine: when a chunked associative
scan accumulates a low-rank precision factor U via concatenation along the
rank axis, the rank can grow without bound. We compress to the top-r left
singular vectors (scaled by their singular values) so that

    Λ_lowrank = U_acc U_acc^T    with    U_acc ∈ R^{d × r}

stays a faithful rank-r approximation of the true accumulated factor.

Approximation budget:
- The combine `[U_acc | U_new]` is exact (concatenation in rank axis); the
  truncation step incurs the standard `‖A − A_r‖_F = sqrt(Σ_{i>r} σ_i^2)`
  error.
- For Kalman info-form on smooth signals the spectrum decays fast so the
  budget is small; tests track `log det Λ_t` for stability.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def streaming_svd_topr(U_acc: Tensor, U_new: Tensor, r: int) -> Tensor:
    """Append `U_new` (rank-r' contribution) to the existing basis `U_acc`
    and return the top-r left singular vectors scaled by singular values.

    Both inputs share leading batch dims `...` and a trailing `(d, *)` shape.

    Args:
      U_acc: [..., d, r1] current basis (may be the empty tensor with r1=0).
      U_new: [..., d, r2] new contribution.
      r:     target rank.

    Returns:
      U_top: [..., d, r] top-r basis (left singular vectors × singular values).
    """
    if U_acc.numel() == 0:
        cat = U_new
    elif U_new.numel() == 0:
        cat = U_acc
    else:
        cat = torch.cat([U_acc, U_new], dim=-1)

    # Compact SVD on the concatenated factor. We discard Vh — the precision
    # contribution UU^T is rotation-invariant on the right side of U.
    Uo, Sg, _Vh = torch.linalg.svd(cat, full_matrices=False)
    r_eff = min(r, Sg.shape[-1])
    return Uo[..., :r_eff] * Sg[..., :r_eff].unsqueeze(-2)


def streaming_svd_topr_aug(
    U_acc: Tensor, U_new: Tensor, r: int
) -> Tuple[Tensor, Tensor]:
    """Brand-style augmentation variant. Returns (U_top, S_top) so callers
    can reuse the singular-value spectrum across calls."""
    if U_acc.numel() == 0:
        Uo, Sg, _ = torch.linalg.svd(U_new, full_matrices=False)
        r_eff = min(r, Sg.shape[-1])
        return Uo[..., :r_eff] * Sg[..., :r_eff].unsqueeze(-2), Sg[..., :r_eff]
    if U_new.numel() == 0:
        Uo, Sg, _ = torch.linalg.svd(U_acc, full_matrices=False)
        r_eff = min(r, Sg.shape[-1])
        return Uo[..., :r_eff] * Sg[..., :r_eff].unsqueeze(-2), Sg[..., :r_eff]
    cat = torch.cat([U_acc, U_new], dim=-1)
    Uo, Sg, _Vh = torch.linalg.svd(cat, full_matrices=False)
    r_eff = min(r, Sg.shape[-1])
    return Uo[..., :r_eff] * Sg[..., :r_eff].unsqueeze(-2), Sg[..., :r_eff]


def reconstruction_error(U_full: Tensor, U_top: Tensor) -> Tensor:
    """Frobenius-norm error ‖U U^T − U_top U_top^T‖_F (averaged over batch).

    Diagnostic only.
    """
    A = U_full @ U_full.transpose(-1, -2)
    A_top = U_top @ U_top.transpose(-1, -2)
    return (A - A_top).flatten(start_dim=-2).norm(dim=-1)
