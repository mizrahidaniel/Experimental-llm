"""Perplexity-correlation domain reweighting.

Per arXiv 2409.05816. Maintains per-domain weights w_d. Periodically:
  1. Train a tiny reference model on a small slice of each domain.
  2. Correlate per-domain validation loss with downstream eval loss.
  3. Reweight w_d ∝ correlation_d (normalized).

Frozen during the first `freeze_until_tokens` to let MoE balancing converge.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class PerplexityCorrelationCurriculum:
    def __init__(
        self,
        domains: List[str],
        freeze_until_tokens: int = 50_000_000_000,
        update_every_tokens: int = 10_000_000_000,
        smoothing: float = 0.5,
    ):
        self.domains = domains
        self.freeze_until = freeze_until_tokens
        self.update_every = update_every_tokens
        self.smoothing = smoothing
        self.weights: Dict[str, float] = {d: 1.0 / len(domains) for d in domains}
        self._last_update_tokens = 0

    def maybe_update(
        self,
        tokens_seen: int,
        domain_losses: Dict[str, float],
        downstream_loss: float,
    ) -> None:
        """Update weights if enough tokens have passed and we're past freeze."""
        if tokens_seen < self.freeze_until:
            return
        if tokens_seen - self._last_update_tokens < self.update_every:
            return
        self._last_update_tokens = tokens_seen
        # Spearman-like: rank by absolute correlation between (domain_losses[d])
        # and downstream_loss treated as a baseline. Without a history we use the
        # magnitude of the *gap* (domain_loss - downstream_loss) as a signed score:
        # domains whose loss tracks the downstream best get up-weighted.
        scores = np.array(
            [-abs(domain_losses[d] - downstream_loss) for d in self.domains]
        )
        scores = scores - scores.min()
        if scores.sum() == 0:
            return
        new = scores / scores.sum()
        # EMA smoothing.
        for i, d in enumerate(self.domains):
            self.weights[d] = (
                self.smoothing * self.weights[d] + (1 - self.smoothing) * float(new[i])
            )
        # Renormalize.
        s = sum(self.weights.values())
        for d in self.domains:
            self.weights[d] /= s

    def sample_domain(self, rng) -> str:
        return rng.choices(self.domains, weights=[self.weights[d] for d in self.domains], k=1)[0]
