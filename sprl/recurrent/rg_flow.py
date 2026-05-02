"""RG-flow regularizer (Bet C).

Adds a penalty term

    L_RG = sum_k || z_{k+1} - T_b z_k ||²

where T_b ∈ R^{d×d} is a *learnable* low-rank-plus-residual operator. The idea
is that the depth recurrence should behave like a renormalization-group block-
spin step in a coarse-grained subspace.

T_b parameterization:
    T_b = U V^T + s · I       (rank r + scaled identity)
with U,V ∈ R^{d × r} and s ∈ R a learnable scalar. Initialized so the spectrum
sits near the Marchenko-Pastur edge (per arXiv 2510.25553's free-probability
prior on universal RG flow).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def marchenko_pastur_init(d: int, r: int, scale: float = 1.0) -> tuple[Tensor, Tensor]:
    """Initialize U, V so U V^T has singular values near the Marchenko-Pastur edge.

    For a d×d matrix sampled from a Gaussian with σ² = 1/d, the bulk of its
    singular values lies in [(1-√(r/d))², (1+√(r/d))²]·σ². We just sample
    Gaussian U, V; this is the standard MP-prior init.
    """
    sigma = scale / math.sqrt(d)
    U = torch.randn(d, r) * sigma
    V = torch.randn(d, r) * sigma
    return U, V


class RGFlowRegularizer(nn.Module):
    def __init__(self, d_model: int, rank: int = 32, init_method: str = "marchenko_pastur"):
        super().__init__()
        self.d = d_model
        self.r = rank
        if init_method == "marchenko_pastur":
            U, V = marchenko_pastur_init(d_model, rank)
        else:
            U = torch.zeros(d_model, rank)
            V = torch.zeros(d_model, rank)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        # Residual scalar — kept small so T_b ≠ I unless trained that way.
        self.s = nn.Parameter(torch.tensor(0.05))

    def matrix(self) -> Tensor:
        eye = torch.eye(self.d, device=self.U.device, dtype=self.U.dtype)
        return self.U @ self.V.t() + self.s * eye

    def step_residual(self, z_prev: Tensor, z_next: Tensor) -> Tensor:
        """Returns ‖z_next − T_b z_prev‖² averaged over batch and seq."""
        # z_prev, z_next: [..., d]
        Tb_z = z_prev @ self.matrix().t()  # [..., d]
        diff = z_next - Tb_z
        return (diff.pow(2).sum(dim=-1)).mean()

    def loss(self, z_iters: list[Tensor]) -> Tensor:
        """L_RG = mean_k ‖z_{k+1} − T_b z_k‖²."""
        if len(z_iters) < 2:
            return torch.tensor(0.0, device=self.U.device)
        losses = [
            self.step_residual(z_iters[k], z_iters[k + 1])
            for k in range(len(z_iters) - 1)
        ]
        return torch.stack(losses).mean()

    def distance_from_identity(self) -> float:
        """Diagnostic: ‖T_b − I‖_F. Kill-criterion threshold ≥ 0.1."""
        with torch.no_grad():
            return float(torch.linalg.norm(self.matrix() - torch.eye(self.d, device=self.U.device)).item())
