"""Active-inference compute-routing.

Allocates the number of recurrence iterations K*(t) per token based on:

    G_t = -log det Λ_t  +  E[predicted next-token loss]
        = epistemic    +   pragmatic

Both signals are produced by the KalmanMemory module (Bet A). When Bet A is
disabled (kalman.enabled = False), `EntropyRouter` falls back to a softmax-
entropy-based proxy so the model is still usable.

A PI controller adjusts the budget Lagrangian κ to drive mean iterations to a
target value.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PIController:
    """Discrete-time PI controller for the compute-budget Lagrangian κ.

    Goal: maintain mean(K*(t)) ≈ k_target. Updates κ using error e = mean_K - k_target.
    """

    def __init__(self, k_p: float = 0.1, k_i: float = 0.01, init_kappa: float = 1.0):
        self.k_p = k_p
        self.k_i = k_i
        self.kappa = init_kappa
        self._integral = 0.0

    def step(self, mean_K: float, k_target: float) -> float:
        e = mean_K - k_target
        self._integral += e
        self.kappa = max(0.0, self.kappa + self.k_p * e + self.k_i * self._integral)
        return self.kappa


class _ScoutHead(nn.Module):
    """Tiny single-layer next-byte predictor used for the pragmatic value estimate.

    Takes h_t [B, S, d] and emits a scalar "predicted loss" per token.
    """

    def __init__(self, d_model: int, hidden: int = 256, vocab: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.vocab = vocab

    def forward(self, h: Tensor) -> Tensor:
        # Predict next-byte distribution; entropy ≈ predicted loss.
        x = F.gelu(self.proj(h))
        logits = self.head(x)  # [B, S, V]
        log_p = F.log_softmax(logits, dim=-1)
        H = -(log_p.exp() * log_p).sum(dim=-1)
        return H


class ActiveInferenceRouter(nn.Module):
    """Maps (epistemic, pragmatic) → integer iteration count K*(t) ∈ [1, K_max].

    The router itself is differentiable with respect to κ via the Lagrangian
    `compute_budget_loss`. The integer K*(t) is produced by `round`+clamp; we
    use a straight-through estimator at training time.
    """

    def __init__(
        self,
        d_model: int,
        k_max: int = 8,
        k_target: int = 4,
        init_kappa: float = 1.0,
    ):
        super().__init__()
        self.k_max = k_max
        self.k_target = k_target
        self.kappa = nn.Parameter(torch.tensor(float(init_kappa)))
        self.scout = _ScoutHead(d_model)
        self.pi = PIController(init_kappa=init_kappa)

    def compute_G(
        self,
        h: Tensor,
        log_det_Lambda: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns the per-token expected free energy G_t. Shape [B, S]."""
        pragmatic = self.scout(h)  # [B, S]
        if log_det_Lambda is None:
            # Without Bet A: epistemic = scout entropy itself (proxy).
            epistemic = pragmatic.detach().clone()
        else:
            # Higher uncertainty (low det Λ) → high epistemic value.
            epistemic = -log_det_Lambda
        return epistemic + pragmatic

    def k_star(self, G: Tensor) -> Tensor:
        """G: [B, S] → K*(t) ∈ {1, …, K_max} as long tensor."""
        # Normalize.
        G_med = G.median(dim=-1, keepdim=True).values
        G_std = G.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        G_norm = (G - G_med) / G_std
        # Soft K with κ.
        soft = 1.0 + self.kappa * G_norm
        soft = soft.clamp(min=1.0, max=float(self.k_max))
        # Round with straight-through.
        hard = soft.detach().round()
        k = (hard - soft).detach() + soft  # STE
        return k

    def compute_budget_loss(self, k_soft: Tensor) -> Tensor:
        """Lagrangian penalty: (mean_t K*(t) − k_target)²."""
        return (k_soft.mean() - self.k_target) ** 2

    def update_kappa(self, mean_K: float) -> None:
        """Update κ via PI controller (called outside the autograd graph)."""
        new = self.pi.step(mean_K, self.k_target)
        with torch.no_grad():
            self.kappa.fill_(new)


class EntropyRouter(nn.Module):
    """Fallback router used when Bet A is disabled.

    Uses the scout-head entropy alone — equivalent to MoR's predicted-loss
    routing. Falsifiable kill experiment 1 demotes ActiveInferenceRouter to
    this if Spearman ρ test fails.
    """

    def __init__(self, d_model: int, k_max: int = 8, k_target: int = 4):
        super().__init__()
        self.k_max = k_max
        self.k_target = k_target
        self.scout = _ScoutHead(d_model)

    def compute_G(self, h: Tensor, **_) -> Tensor:
        return self.scout(h)

    def k_star(self, G: Tensor) -> Tensor:
        # Linear mapping from per-token G into the range [1, K_max].
        G_min = G.min(dim=-1, keepdim=True).values
        G_max = G.max(dim=-1, keepdim=True).values.clamp_min(G_min + 1e-6)
        norm = (G - G_min) / (G_max - G_min)
        soft = 1.0 + (self.k_max - 1) * norm
        hard = soft.detach().round()
        return (hard - soft).detach() + soft
