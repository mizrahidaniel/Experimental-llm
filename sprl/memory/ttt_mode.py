"""Test-time training (TTT) mode for the Kalman info-form memory.

Per Titans (arXiv 2501.00663) and MIRAS (arXiv 2504.13173), the belief memory
can be made test-time-adaptive by maintaining a state across inference calls
and nudging the local transition T_θ via a small auxiliary residual driven by
the surprise signal.

This module exposes:

  * TestTimeKalmanState — a stateful object holding (η_t, D_t, U_t) and the
    accumulated surprise scalar; advanced one step at a time via `update()`.

  * `update(memory, state, h)` — single-step update using
    `KalmanInfoMemory.streaming_step`, plus surprise tracking.

  * `surprise_residual(memory, state, h, lr)` — small content-residual the
    caller can add to h to drive it toward the memory mean μ in proportion
    to accumulated surprise. This is the test-time T_θ nudge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from sprl.memory.kalman_info import KalmanInfoMemory
from sprl.memory.low_rank_precision import smw_inverse_apply


@dataclass
class TestTimeKalmanState:
    """Streaming state for a TTT session.

    Holds the most recent (η, D, U) tuple plus a running surprise scalar
    `cum_surprise` that tracks Σ_t -log p(h_t | state_{<t}). Surprise is
    bounded below by 0 (relative).
    """

    # Tell pytest not to collect this dataclass as a test class.
    __test__ = False

    eta: Optional[Tensor] = None
    D: Optional[Tensor] = None
    U: Optional[Tensor] = None
    cum_surprise: float = 0.0
    n_steps: int = 0
    history_eta: List[Tensor] = field(default_factory=list)

    def state_tuple(self) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        if self.eta is None:
            return None
        return self.eta, self.D, self.U

    def reset(self) -> None:
        self.eta = None
        self.D = None
        self.U = None
        self.cum_surprise = 0.0
        self.n_steps = 0
        self.history_eta.clear()


def update(
    memory: KalmanInfoMemory,
    state: TestTimeKalmanState,
    h_new: Tensor,
    record: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Advance `state` by one observation step.

    Args:
      memory: the KalmanInfoMemory module (parameters are read; not updated).
      state: TestTimeKalmanState — modified in place.
      h_new: [B, d_model] new token embedding.
      record: if True, append eta to state.history_eta (diagnostic).

    Returns:
      The updated (eta, D, U) triple.
    """
    prev = state.state_tuple()
    eta, D, U = memory.streaming_step(h_new, prev)
    state.eta, state.D, state.U = eta, D, U
    state.n_steps += 1

    # Surprise = innovation energy under the (current) precision diagonal.
    if prev is not None:
        with torch.no_grad():
            mu = smw_inverse_apply(D, U, eta)
            innov = h_new - mu
            energy = (innov * D * innov).sum(dim=-1).mean().item()
            state.cum_surprise += float(energy)
    if record:
        state.history_eta.append(eta.detach())
    return eta, D, U


def surprise_residual(
    memory: KalmanInfoMemory,
    state: TestTimeKalmanState,
    h_new: Tensor,
    lr: float = 1.0e-3,
) -> Tensor:
    """Return a small residual update to add to h_new based on accumulated
    surprise — the test-time T_θ nudge.

    This is a *content* residual (modifies the input embedding, not the
    parameters), so the autograd graph for the running model stays small.
    """
    if state.eta is None:
        return torch.zeros_like(h_new)
    with torch.no_grad():
        mu = smw_inverse_apply(state.D, state.U, state.eta)
        innov = h_new - mu
        scale = lr * min(1.0, state.cum_surprise / max(1.0, state.n_steps))
    return -scale * innov
