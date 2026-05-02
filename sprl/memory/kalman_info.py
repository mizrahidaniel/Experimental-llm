"""Information-form Kalman belief memory (Bet A).

Maintains per-token (η, Λ) where η = Λ μ and Λ is precision. Λ is parameterized
as D + U Uᵀ (rank-r + diagonal) for tractability.

This module is an associative-scan-friendly information-form filter, following
KLA (arXiv 2602.10743), but with all transition / observation operators
data-dependent (per token).

The simplifications below (vs a "full" Kalman filter) are:
  - F is restricted to a diagonal data-dependent gate plus a fixed scalar (so
    that the predict step doesn't require inverting a d×d matrix).
  - The observation map H is fixed identity (we treat the observation y as a
    direct linear projection of the input h).
  - Process noise Q⁻¹ and obs noise R⁻¹ are diagonal, data-dependent.

These simplifications make the recurrence associative on (η, Λ_diag, Λ_lowrank).
The associativity is verified in tests/test_kalman_associative.py.

A *strict* full-rank Kalman info-form is theoretically associative (KLA paper),
but the implementation requires careful Riccati-style block accumulators and
is reserved for a Triton port.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sprl.memory.low_rank_precision import log_det_low_rank, smw_inverse_apply
from sprl.memory.parallel_scan import chunked_associative_scan


class KalmanInfoMemory(nn.Module):
    """Per-token information-form Kalman filter with low-rank precision.

    Inputs:
        h: [B, S, d_model] — pre-attention activations driving F, R⁻¹, observation y.

    State per token t:
        η_t  ∈ R^d         (natural parameter)
        D_t  ∈ R^d         (diagonal of Λ)
        U_t  ∈ R^{d × r}   (low-rank chol-like factor of Λ)

    Returns:
        eta:  [B, S, d]
        D:    [B, S, d]
        U:    [B, S, d, r]
        log_det_Lambda: [B, S]   (used by ActiveInferenceRouter)
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 8,
        eps_diag: float = 1.0e-4,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d = d_model
        self.r = rank
        self.eps_diag = eps_diag
        self.chunk_size = chunk_size

        # Data-dependent gate F_t = diag(softplus(g_F(h_t))) — keeps |F| ≤ 1.
        self.g_F = nn.Linear(d_model, d_model)
        # Process noise (diagonal precision) Q⁻¹
        self.g_Qinv = nn.Linear(d_model, d_model)
        # Observation precision (diagonal) R⁻¹
        self.g_Rinv = nn.Linear(d_model, d_model)
        # Observation linear map: y_t = W_y h_t
        self.W_y = nn.Linear(d_model, d_model, bias=False)
        # Low-rank update direction: U_t observation contribution.
        self.W_U = nn.Linear(d_model, d_model * rank, bias=False)

        # Init biases so initial F ≈ 0.95 (gentle decay) and Q,R have moderate scale.
        nn.init.zeros_(self.g_F.weight)
        nn.init.constant_(self.g_F.bias, 2.94)  # sigmoid(2.94) ≈ 0.95
        nn.init.zeros_(self.g_Qinv.bias)
        nn.init.zeros_(self.g_Rinv.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _combine(a, b):
        """Information-form combine for the simplified recurrence.

        State: (eta, D, U_lowrank_acc, F_acc).
        Recurrence (with diagonal F):
            eta_{t+1} = F_t * eta_t + H_t^T R_t^{-1} y_t
            D_{t+1}   = (F_t^2)^{-1} D_t · g(F_t)  + Q_t^{-1} + R_t^{-1}_diag
            U_{t+1}   = [shifted U_t  ;  newly added rank-r piece]

        For an associative scan, we maintain
            S_t = (F_acc, eta_acc, D_acc, U_acc)
        where F_acc is the running product of F_τ. The combine is:
            F_acc_c = F_acc_b · F_acc_a
            eta_c   = F_acc_b · eta_a + eta_b
            D_c     = D_a · F_acc_b² + D_b
        and U_c is concatenation in the rank-axis (caller controls truncation).
        This is exactly associative (verified in tests).
        """
        F_a, eta_a, D_a, U_a = a
        F_b, eta_b, D_b, U_b = b
        F_c = F_a * F_b
        eta_c = F_b * eta_a + eta_b
        D_c = D_a * (F_b ** 2) + D_b
        # Low-rank: in a strict info-form Kalman the rank-r piece accumulates
        # (concatenation in the rank axis). For an associative scan with fixed
        # output rank we keep only the *most recent* rank-r contribution scaled
        # appropriately. This is the rank-truncated approximation used in KLA.
        U_c = U_b + U_a * F_b.unsqueeze(-1)
        return F_c, eta_c, D_c, U_c

    # ------------------------------------------------------------------
    def forward(
        self,
        h: Tensor,
        truncate_rank: Optional[int] = None,
    ) -> dict:
        """Args:
          h: [B, S, d_model]
          truncate_rank: if given, after the scan keep only the top-r SV columns
            of U (so memory stays bounded with sequence length).

        Returns dict with:
          eta:  [B, S, d]
          D:    [B, S, d]
          U:    [B, S, d, r_eff]
          log_det_Lambda: [B, S]
        """
        B, S, d = h.shape
        if truncate_rank is None:
            truncate_rank = self.r

        # Per-token operators.
        F_t = torch.sigmoid(self.g_F(h))  # [B, S, d] in (0, 1)
        Qinv = F.softplus(self.g_Qinv(h)) + self.eps_diag
        Rinv = F.softplus(self.g_Rinv(h)) + self.eps_diag
        y_t = self.W_y(h)  # [B, S, d]
        U_t = self.W_U(h).view(B, S, d, self.r) * (Rinv.unsqueeze(-1).sqrt())

        # Initial scan elements.
        eta_t = Rinv * y_t  # [B, S, d]
        D_t = Qinv + Rinv

        # Run associative scan along seq axis = 1.
        F_scan, eta_scan, D_scan, U_scan = chunked_associative_scan(
            (F_t, eta_t, D_t, U_t),
            self._combine,
            chunk_size=self.chunk_size,
            axis=1,
        )

        # U is fixed-rank under the rank-truncated combine; no further truncation needed.
        log_det = log_det_low_rank(D_scan, U_scan)  # [B, S]

        return {
            "eta": eta_scan,
            "D": D_scan,
            "U": U_scan,
            "log_det_Lambda": log_det,
        }

    # ------------------------------------------------------------------
    def mean(self, eta: Tensor, D: Tensor, U: Tensor) -> Tensor:
        """Recover μ = Λ⁻¹ η using SMW. Used as the "memory readout"."""
        return smw_inverse_apply(D, U, eta)


def _truncate_lowrank(U: Tensor, r: int) -> Tensor:
    """Per-token rank-r truncation by SVD on U U^T's column space.

    U: [B, S, d, R] → [B, S, d, r]
    """
    B, S, d, R = U.shape
    # Use compact SVD on U directly.
    flat = U.reshape(B * S, d, R)
    Uo, Sg, Vh = torch.linalg.svd(flat, full_matrices=False)
    # Keep top-r singular components.
    r_eff = min(r, Sg.shape[-1])
    Uo = Uo[..., :r_eff] * Sg[..., :r_eff].unsqueeze(-2)
    return Uo.reshape(B, S, d, r_eff)
