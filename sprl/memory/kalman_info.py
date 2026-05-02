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

Two combine modes:
  - `strict_rank=True` (default): low-rank factor U accumulates via *concat*
    in the rank axis; after each chunk in the chunked scan we project to
    top-r via streaming SVD (Brand 2002). KLA-style strict info-form filter;
    associative under the (concat, then-truncate) composition rule applied
    at chunk boundaries.
  - `strict_rank=False`: legacy combine `U_c = U_b + F_b · U_a` — bit-
    equivalent to the original implementation, kept for ablations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sprl.memory.low_rank_precision import log_det_low_rank, smw_inverse_apply
from sprl.memory.parallel_scan import chunked_associative_scan
from sprl.memory.streaming_svd import streaming_svd_topr


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
        strict_rank: bool = True,
    ):
        super().__init__()
        self.d = d_model
        self.r = rank
        self.eps_diag = eps_diag
        self.chunk_size = chunk_size
        self.strict_rank = strict_rank

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
        """Information-form combine — legacy rank-truncated variant.

        State: (F_acc, eta_acc, D_acc, U_acc). Combine:
            F_c   = F_a · F_b
            eta_c = F_b · eta_a + eta_b
            D_c   = D_a · F_b² + D_b
            U_c   = U_b + F_b · U_a       (rank-truncated KLA approx)

        Exactly associative; verified in tests. Used when `strict_rank=False`.
        """
        F_a, eta_a, D_a, U_a = a
        F_b, eta_b, D_b, U_b = b
        F_c = F_a * F_b
        eta_c = F_b * eta_a + eta_b
        D_c = D_a * (F_b ** 2) + D_b
        U_c = U_b + U_a * F_b.unsqueeze(-1)
        return F_c, eta_c, D_c, U_c

    # ------------------------------------------------------------------
    @staticmethod
    def _combine_strict_concat(a, b):
        """Strict-rank info-form combine: U accumulates by concatenation.

        State: (F_acc, eta_acc, D_acc, U_acc) — U_acc has a *growing* rank
        axis. Caller compresses it back to fixed rank-r via `streaming_svd_topr`
        at chunk boundaries; the (concat, then-truncate) composition rule
        remains associative for Λ_low = U U^T.

        Combine:
            F_c   = F_a · F_b
            eta_c = F_b · eta_a + eta_b
            D_c   = D_a · F_b² + D_b
            U_c   = [F_b · U_a | U_b]
        """
        F_a, eta_a, D_a, U_a = a
        F_b, eta_b, D_b, U_b = b
        F_c = F_a * F_b
        eta_c = F_b * eta_a + eta_b
        D_c = D_a * (F_b ** 2) + D_b
        U_a_rot = U_a * F_b.unsqueeze(-1)
        U_c = torch.cat([U_a_rot, U_b], dim=-1)
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

        if self.strict_rank:
            F_scan, eta_scan, D_scan, U_scan = _strict_chunked_scan(
                (F_t, eta_t, D_t, U_t),
                chunk_size=self.chunk_size,
                axis=1,
                target_rank=truncate_rank,
            )
        else:
            F_scan, eta_scan, D_scan, U_scan = chunked_associative_scan(
                (F_t, eta_t, D_t, U_t),
                self._combine,
                chunk_size=self.chunk_size,
                axis=1,
            )

        # U is fixed-rank under the streaming-SVD truncation.
        log_det = log_det_low_rank(D_scan, U_scan)  # [B, S]

        return {
            "eta": eta_scan,
            "D": D_scan,
            "U": U_scan,
            "log_det_Lambda": log_det,
        }

    # ------------------------------------------------------------------
    def streaming_step(
        self,
        h_new: Tensor,
        state: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Single-token streaming update for inference / TTT mode.

        Advances (η, D, U) by exactly one observation step using the same
        recurrence as `forward`, but without the chunked associative scan
        (better suited for autoregressive generation).

        Args:
          h_new: [B, d_model] observation for the new token.
          state: previous (eta, D, U) tuple, or None for the initial step.

        Returns the updated (eta, D, U) state.
        """
        B, d = h_new.shape
        F_t = torch.sigmoid(self.g_F(h_new))  # [B, d]
        Qinv = F.softplus(self.g_Qinv(h_new)) + self.eps_diag
        Rinv = F.softplus(self.g_Rinv(h_new)) + self.eps_diag
        y_t = self.W_y(h_new)
        U_t = self.W_U(h_new).view(B, d, self.r) * (Rinv.unsqueeze(-1).sqrt())

        eta_obs = Rinv * y_t
        D_obs = Qinv + Rinv

        if state is None:
            return eta_obs, D_obs, U_t

        eta_p, D_p, U_p = state
        eta = F_t * eta_p + eta_obs
        D = D_p * (F_t ** 2) + D_obs
        if self.strict_rank:
            U_rot = U_p * F_t.unsqueeze(-1)
            U_full = torch.cat([U_rot, U_t], dim=-1)
            empty = torch.zeros(
                *U_full.shape[:-1], 0, dtype=U_full.dtype, device=U_full.device
            )
            U = streaming_svd_topr(empty, U_full, self.r)
        else:
            U = U_t + U_p * F_t.unsqueeze(-1)
        return eta, D, U

    # ------------------------------------------------------------------
    def mean(self, eta: Tensor, D: Tensor, U: Tensor) -> Tensor:
        """Recover μ = Λ⁻¹ η using SMW. Used as the "memory readout"."""
        return smw_inverse_apply(D, U, eta)


# ---------------------------------------------------------------------------


def _strict_associative_scan_chunk(
    elements: Tuple[Tensor, Tensor, Tensor, Tensor],
    target_rank: int,
    axis: int = 1,
) -> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Sequential strict-rank scan over a single chunk.

    Within the chunk we accumulate U at the *growing* rank (each step adds
    the new rank-r contribution by concatenation). This is exact. Per-step
    output U is then compressed via streaming SVD to a fixed target_rank
    tensor (so all per-step outputs share a rank axis and can be stacked).

    Returns ((F_out, eta_out, D_out, U_out), carry) where carry is the raw
    (uncompressed) accumulator state at the end of the chunk.
    """
    permuted = tuple(t.movedim(axis, 0) for t in elements)
    F_p, eta_p, D_p, U_p = permuted
    T = F_p.shape[0]

    F_out = [F_p[0]]
    eta_out = [eta_p[0]]
    D_out = [D_p[0]]
    U_out = [U_p[0]]

    F_acc, eta_acc, D_acc, U_acc = F_p[0], eta_p[0], D_p[0], U_p[0]
    for t in range(1, T):
        F_n, eta_n, D_n, U_n = F_p[t], eta_p[t], D_p[t], U_p[t]
        F_acc = F_acc * F_n
        eta_acc = F_n * eta_acc + eta_n
        D_acc = D_acc * (F_n ** 2) + D_n
        U_acc = torch.cat([U_acc * F_n.unsqueeze(-1), U_n], dim=-1)
        F_out.append(F_acc)
        eta_out.append(eta_acc)
        D_out.append(D_acc)
        U_out.append(U_acc)

    U_out_truncated = []
    for u in U_out:
        empty = torch.zeros(*u.shape[:-1], 0, dtype=u.dtype, device=u.device)
        U_out_truncated.append(streaming_svd_topr(empty, u, target_rank))

    F_stack = torch.stack(F_out, dim=0).movedim(0, axis)
    eta_stack = torch.stack(eta_out, dim=0).movedim(0, axis)
    D_stack = torch.stack(D_out, dim=0).movedim(0, axis)
    U_stack = torch.stack(U_out_truncated, dim=0).movedim(0, axis)
    carry = (F_acc, eta_acc, D_acc, U_acc)
    return (F_stack, eta_stack, D_stack, U_stack), carry


def _strict_chunked_scan(
    elements: Tuple[Tensor, Tensor, Tensor, Tensor],
    chunk_size: int,
    axis: int,
    target_rank: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Chunked strict-rank scan with streaming SVD compression at chunk
    boundaries."""
    permuted = tuple(t.movedim(axis, 0) for t in elements)
    F_p, eta_p, D_p, U_p = permuted
    T = F_p.shape[0]

    out_chunks = []
    carry: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        sub = (F_p[s:e], eta_p[s:e], D_p[s:e], U_p[s:e])
        if carry is not None:
            F_c, eta_c, D_c, U_c = carry
            empty = torch.zeros(*U_c.shape[:-1], 0, dtype=U_c.dtype, device=U_c.device)
            U_c_compressed = streaming_svd_topr(empty, U_c, target_rank)
            sub_with_carry = (
                torch.cat([F_c.unsqueeze(0), sub[0]], dim=0),
                torch.cat([eta_c.unsqueeze(0), sub[1]], dim=0),
                torch.cat([D_c.unsqueeze(0), sub[2]], dim=0),
                torch.cat([U_c_compressed.unsqueeze(0), sub[3]], dim=0),
            )
            (F_o, eta_o, D_o, U_o), carry = _strict_associative_scan_chunk(
                tuple(x.movedim(0, axis) for x in sub_with_carry),
                target_rank=target_rank,
                axis=axis,
            )
            F_o = F_o.movedim(axis, 0)[1:].movedim(0, axis)
            eta_o = eta_o.movedim(axis, 0)[1:].movedim(0, axis)
            D_o = D_o.movedim(axis, 0)[1:].movedim(0, axis)
            U_o = U_o.movedim(axis, 0)[1:].movedim(0, axis)
        else:
            (F_o, eta_o, D_o, U_o), carry = _strict_associative_scan_chunk(
                tuple(x.movedim(0, axis) for x in sub),
                target_rank=target_rank,
                axis=axis,
            )
        out_chunks.append((F_o, eta_o, D_o, U_o))

    F_cat = torch.cat([c[0] for c in out_chunks], dim=axis)
    eta_cat = torch.cat([c[1] for c in out_chunks], dim=axis)
    D_cat = torch.cat([c[2] for c in out_chunks], dim=axis)
    U_cat = torch.cat([c[3] for c in out_chunks], dim=axis)
    return F_cat, eta_cat, D_cat, U_cat


def _truncate_lowrank(U: Tensor, r: int) -> Tensor:
    """Per-token rank-r truncation by SVD on U U^T's column space.

    U: [B, S, d, R] → [B, S, d, r]
    """
    B, S, d, R = U.shape
    flat = U.reshape(B * S, d, R)
    Uo, Sg, Vh = torch.linalg.svd(flat, full_matrices=False)
    r_eff = min(r, Sg.shape[-1])
    Uo = Uo[..., :r_eff] * Sg[..., :r_eff].unsqueeze(-2)
    return Uo.reshape(B, S, d, r_eff)
