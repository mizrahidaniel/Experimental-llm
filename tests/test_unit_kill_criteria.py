"""Kill-criterion harness — checks that the *measurement* code is correct.

These tests don't actually run the kill experiments at scale; they verify that
the diagnostic code correctly detects pass and fail conditions when fed
synthetic data with a known answer.
"""

import torch

from sprl.recurrent.rg_flow import RGFlowRegularizer
from sprl.utils.diagnostics import (
    operator_distance_from_identity,
    power_law_fit,
    spearman_rho,
)


def test_kill_A_synthetic_pass():
    """Bet A's kill metric is ρ(uncertainty, surprise) ≥ 0.5 with uncertainty = -log_det Λ.

    Λ is a precision matrix; high log_det ⇒ low uncertainty ⇒ low surprise. The
    kill criterion correlates the *signed* uncertainty against surprise and
    expects positive correlation.
    """
    surprise = torch.linspace(0, 1, 100)
    log_det = -surprise + 0.01 * torch.randn(100)  # synthetic: planted anti-corr
    uncertainty = (-log_det).tolist()
    rho = spearman_rho(uncertainty, surprise.tolist())
    # uncertainty should track surprise positively.
    assert rho > 0.9
    assert rho >= 0.5  # passes Bet A's signed kill criterion


def test_kill_A_synthetic_fail():
    """Random unrelated signals fail."""
    a = torch.randn(100).tolist()
    b = torch.randn(100).tolist()
    rho = spearman_rho(a, b)
    assert abs(rho) < 0.5  # fails kill criterion (which is correct behavior here)


def test_kill_C_synthetic_power_law_pass():
    iters = list(range(1, 50))
    losses = [3.0 * (i ** -0.7) for i in iters]
    _, b, r2 = power_law_fit(iters, losses)
    assert r2 > 0.9
    assert abs(b - 0.7) < 1e-3


def test_kill_C_synthetic_power_law_fail_for_constant():
    iters = list(range(1, 30))
    losses = [1.0] * len(iters)
    _, _, r2 = power_law_fit(iters, losses)
    # For constant signal we don't even meet the variance criterion; r² should be tiny.
    assert r2 < 0.5


def test_kill_C_T_b_distance_passes_at_init():
    rg = RGFlowRegularizer(d_model=32, rank=4)
    dist = operator_distance_from_identity(rg.matrix())
    assert dist >= 0.1  # spec's minimum non-trivial threshold


def test_kill_C_T_b_distance_fails_when_at_identity():
    eye = torch.eye(8)
    dist = operator_distance_from_identity(eye)
    assert dist < 0.1
