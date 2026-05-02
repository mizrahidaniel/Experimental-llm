"""Auxiliary-loss-free expert balancing.

Per DeepSeek-V3 (arXiv 2412.19437) and the primal-dual analysis in arXiv
2512.03915. Tracks per-expert load and nudges the router *bias* toward the
mean load — no gradient signal, just a per-step bias update.

Optional small auxiliary CE loss is exposed via `aux_loss(logits, indices)` so
callers can mix in a backup balance term (weight ~0.001) during the curriculum
phase, where stationarity assumptions are violated.
"""

from __future__ import annotations

import torch
from torch import Tensor


class AuxLossFreeBalancer:
    """Stateless update rule. Owns the bias tensor in the router."""

    def __init__(self, n_experts: int, lr: float = 1.0e-3):
        self.n_experts = n_experts
        self.lr = lr

    @torch.no_grad()
    def update(
        self,
        router_bias: Tensor,
        topk_indices: Tensor,
    ) -> None:
        """Adjust router_bias in-place based on routed counts.

        topk_indices: [N, k] of expert ids.
        """
        counts = torch.bincount(
            topk_indices.flatten(), minlength=self.n_experts
        ).float()
        mean = counts.mean()
        # Underused experts (counts < mean) get a positive nudge.
        delta = -self.lr * (counts - mean) / mean.clamp_min(1.0)
        router_bias.add_(delta.to(router_bias.dtype).to(router_bias.device))

    @staticmethod
    def aux_loss(full_logits: Tensor, topk_indices: Tensor, n_experts: int) -> Tensor:
        """Small CE-style backup balance loss.

        Implements the standard ‘fraction tokens to expert × routing prob’ term
        from the original Switch / DeepSeek MoE.
        """
        N = full_logits.shape[0]
        probs = full_logits.softmax(dim=-1)  # [N, E]
        f = torch.zeros(n_experts, device=full_logits.device, dtype=full_logits.dtype)
        f.scatter_add_(
            0,
            topk_indices.flatten(),
            torch.ones_like(topk_indices.flatten(), dtype=full_logits.dtype),
        )
        f = f / max(N * topk_indices.shape[1], 1)
        P = probs.mean(dim=0)  # [E]
        return (f * P).sum() * n_experts
