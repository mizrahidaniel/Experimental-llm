"""Low-rank-plus-diagonal precision algebra.

Λ = D + U Uᵀ      with D ∈ R^d (diagonal) and U ∈ R^{d × r}, r ≪ d.

Uses the Sherman-Morrison-Woodbury identity for inverses and the matrix
determinant lemma for log-determinants. All operations are batched over
arbitrary leading dims.
"""

from __future__ import annotations

import torch
from torch import Tensor

EPS = 1.0e-6


def smw_inverse_apply(D: Tensor, U: Tensor, x: Tensor) -> Tensor:
    """Compute (D + U Uᵀ)⁻¹ x without forming a d×d matrix.

    Identity:
        (D + UUᵀ)⁻¹ = D⁻¹ − D⁻¹ U (I + Uᵀ D⁻¹ U)⁻¹ Uᵀ D⁻¹

    D: [..., d] (diagonal of D); U: [..., d, r]; x: [..., d]
    Returns: [..., d]
    """
    Dinv = 1.0 / D.clamp_min(EPS)
    Dinv_x = Dinv * x
    DinvU = U * Dinv.unsqueeze(-1)  # [..., d, r]
    M = torch.einsum("...dr,...ds->...rs", U, DinvU)  # [..., r, r]
    r = U.shape[-1]
    eye = torch.eye(r, device=U.device, dtype=U.dtype).expand_as(M)
    M = M + eye
    UTDinvx = torch.einsum("...dr,...d->...r", U, Dinv_x)
    sol = torch.linalg.solve(M, UTDinvx.unsqueeze(-1)).squeeze(-1)  # [..., r]
    correction = torch.einsum("...dr,...r->...d", DinvU, sol)
    return Dinv_x - correction


def smw_inverse(D: Tensor, U: Tensor) -> tuple[Tensor, Tensor]:
    """Return (D̃, Ũ) such that D̃ + Ũ Ũᵀ = (D + UUᵀ)⁻¹.

    Note: the inverse of a rank-r perturbation of a diagonal is *not*
    in general of the same rank-r-plus-diagonal form. We return the
    closest such factorization for downstream low-rank ops:

        (D + UUᵀ)⁻¹ = D⁻¹ − (D⁻¹ U) M⁻¹ (D⁻¹ U)ᵀ

    so D̃ = D⁻¹, and Ũ = (D⁻¹ U) · L^{-T} where L Lᵀ = M = I + Uᵀ D⁻¹ U.
    The sign on the rank correction is negative; we encode that by returning
    Ũ = i·(...)·, but PyTorch can't store imaginary easily. So we return
    (D̃, Ũ) and a sign flag '-1' meaning the rank piece subtracts.

    Provided here for completeness; downstream prefers `smw_inverse_apply`.
    """
    Dinv = 1.0 / D.clamp_min(EPS)
    DinvU = U * Dinv.unsqueeze(-1)
    M = torch.einsum("...dr,...ds->...rs", U, DinvU)
    r = U.shape[-1]
    eye = torch.eye(r, device=U.device, dtype=U.dtype).expand_as(M)
    M = M + eye
    L = torch.linalg.cholesky(M)
    L_invT = torch.linalg.solve_triangular(
        L.transpose(-1, -2), torch.eye(r, device=U.device, dtype=U.dtype).expand_as(L),
        upper=True,
    )
    U_tilde = torch.einsum("...dr,...rs->...ds", DinvU, L_invT)
    return Dinv, U_tilde  # interpret as (D⁻¹) − Ũ Ũᵀ


def log_det_low_rank(D: Tensor, U: Tensor) -> Tensor:
    """log det(D + U Uᵀ) via the matrix determinant lemma.

    log det(D + UUᵀ) = log det(D) + log det(I_r + Uᵀ D⁻¹ U)
    """
    eps = EPS
    D_safe = D.clamp_min(eps)
    log_det_D = torch.log(D_safe).sum(dim=-1)
    DinvU = U / D_safe.unsqueeze(-1)
    M = torch.einsum("...dr,...ds->...rs", U, DinvU)
    r = U.shape[-1]
    eye = torch.eye(r, device=U.device, dtype=U.dtype).expand_as(M)
    sign, logabsdet = torch.linalg.slogdet(eye + M)
    return log_det_D + logabsdet
