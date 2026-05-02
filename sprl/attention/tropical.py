"""Tropical (max-plus) attention head — Bet B.

Replaces softmax with max in the attention formula:
    standard:  y_i = sum_j softmax(s_ij) V_j
    tropical:  y_i_d = max_j (s_ij + V_jd)
where s_ij = max_d (q_id + k_jd) is the tropical inner product.

Implements:
1. Pure max-plus forward (`tropical_attn_forward`).
2. Softmax-with-large-β surrogate (`soft_tropical_attn_forward`) for the
   warmup period — anneals β from beta_min → beta_max so gradients are smooth.
3. `build_dual_algebra_attn(...)` — head-partitioned wrapper that runs the
   first `frac` heads tropical and the rest softmax, plumbed into MLA.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Pure max-plus
# ---------------------------------------------------------------------------


def tropical_inner(Q: Tensor, K: Tensor) -> Tensor:
    """s_ij = max_d (q_id + k_jd).
    Q: [..., Sq, d]; K: [..., Sk, d] → s: [..., Sq, Sk]
    """
    # Pairwise sum [..., Sq, Sk, d], max over d.
    return (Q.unsqueeze(-2) + K.unsqueeze(-3)).amax(dim=-1)


def tropical_attn_forward(
    Q: Tensor, K: Tensor, V: Tensor, attn_mask: Optional[Tensor] = None
) -> Tensor:
    """Pure max-plus attention.

    Q,K: [B, H, S, d_qk]; V: [B, H, S, d_v] → [B, H, S, d_v].
    Causal masking: caller must add a mask of -inf to disallowed positions.
    """
    s = tropical_inner(Q, K)  # [B, H, Sq, Sk]
    if attn_mask is not None:
        s = s + attn_mask  # -inf at disallowed positions

    # y_i_d = max_j (s_ij + V_jd).
    # Broadcast s over d, V over Sq.
    pair = s.unsqueeze(-1) + V.unsqueeze(-3)  # [B, H, Sq, Sk, d_v]
    return pair.amax(dim=-2)  # [B, H, Sq, d_v]


# ---------------------------------------------------------------------------
# Softmax-with-large-β surrogate
# ---------------------------------------------------------------------------


def soft_tropical_attn_forward(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    beta: float,
    attn_mask: Optional[Tensor] = None,
) -> Tensor:
    """Smooth surrogate of tropical attention via log-sum-exp at temperature 1/β.

    Hard tropical:  y_id = max_j (s_ij + V_jd),  s_ij = max_d (q_id + k_jd).
    Soft surrogate: y_id = (1/β) · logsumexp_j ( β·(s_ij + V_jd) ),
                    s_ij = (1/β) · logsumexp_d ( β·(q_id + k_jd) ).
    As β → ∞, both `max`s recover and the surrogate equals the hard form.
    """
    # Tropical inner via logsumexp over d.
    pair_qk = Q.unsqueeze(-2) + K.unsqueeze(-3)  # [..., Sq, Sk, d_qk]
    s = (1.0 / beta) * torch.logsumexp(beta * pair_qk, dim=-1)  # [..., Sq, Sk]
    if attn_mask is not None:
        s = s + attn_mask

    # Outer "max" over Sk for each (i, d): logsumexp(β·(s_ij + V_jd)) over j.
    # Shape: [..., Sq, Sk, d_v]
    pair_sV = s.unsqueeze(-1) + V.unsqueeze(-3)
    out = (1.0 / beta) * torch.logsumexp(beta * pair_sV, dim=-2)
    return out


# ---------------------------------------------------------------------------
# Head-partitioned dual-algebra
# ---------------------------------------------------------------------------


class TropicalAttentionHead(nn.Module):
    """Pluggable kernel for MLA's `head_attn_fn`.

    On the first `n_tropical_heads` heads, runs (soft) tropical attention with
    current β. On the remaining heads, runs standard softmax. Returns a
    fused [B, H, S, d_h] tensor.
    """

    def __init__(
        self,
        n_heads: int,
        fraction_tropical: float = 0.25,
        beta_init: float = 1.0,
        causal: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_tropical = max(1, int(round(n_heads * fraction_tropical)))
        self.causal = causal
        # Buffer so β rides on .to(device).
        self.register_buffer("beta", torch.tensor(beta_init), persistent=False)

    def set_beta(self, beta: float) -> None:
        self.beta.fill_(beta)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Q,K: [B, H, S, d_qk]; V: [B, H, S, d_v]
        B, H, S, _ = Q.shape
        d_v = V.shape[-1]
        d_qk = Q.shape[-1]
        scale = d_qk ** -0.5

        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=Q.device, dtype=Q.dtype),
                diagonal=1,
            )

        full_mask = causal_mask
        if attn_mask is not None:
            full_mask = (
                attn_mask if full_mask is None else (full_mask + attn_mask)
            )

        # Tropical heads: 0..n_tropical
        Qt = Q[:, : self.n_tropical]
        Kt = K[:, : self.n_tropical]
        Vt = V[:, : self.n_tropical]
        out_t = soft_tropical_attn_forward(Qt, Kt, Vt, beta=float(self.beta), attn_mask=full_mask)

        # Softmax heads.
        Qs = Q[:, self.n_tropical:]
        Ks = K[:, self.n_tropical:]
        Vs = V[:, self.n_tropical:]
        scores = torch.matmul(Qs, Ks.transpose(-2, -1)) * scale
        if full_mask is not None:
            scores = scores + full_mask
        out_s = torch.matmul(torch.softmax(scores, dim=-1), Vs)

        return torch.cat([out_t, out_s], dim=1)


def build_dual_algebra_attn(
    n_heads: int,
    fraction_tropical: float,
    causal: bool = True,
) -> Callable[[Tensor, Tensor, Tensor, Optional[Tensor]], Tensor]:
    """Convenience: returns a closure suitable for `MLA.head_attn_fn`."""
    head = TropicalAttentionHead(n_heads, fraction_tropical, causal=causal)

    def fn(Q, K, V, mask):
        return head(Q, K, V, mask)

    fn.module = head  # type: ignore[attr-defined]
    return fn
