"""Bet A's kill metric is signed: ρ(uncertainty, surprise) ≥ 0.5 with
uncertainty = -log_det Λ. The router itself uses the same sign in compute_G.
"""

import torch

from sprl.recurrent.active_inference_router import ActiveInferenceRouter
from sprl.utils.diagnostics import spearman_rho


def test_uncertainty_is_negative_log_det():
    """compute_G must use -log_det_Lambda (high precision ⇒ low epistemic value)."""
    r = ActiveInferenceRouter(d_model=16, k_max=4, k_target=2)
    h = torch.randn(1, 8, 16)
    log_det_high_certainty = torch.full((1, 8), 10.0)   # high precision
    log_det_low_certainty = torch.full((1, 8), -10.0)   # low precision
    G_high = r.compute_G(h, log_det_high_certainty)
    G_low = r.compute_G(h, log_det_low_certainty)
    # Low precision ⇒ high uncertainty ⇒ higher G (more iterations).
    assert (G_low > G_high).all(), \
        "uncertainty signal is inverted: low precision should produce higher G"


def test_kill_metric_signed_anticorrelation():
    """If log_det Λ anti-correlates with surprise (the desired regime),
    ρ(-log_det Λ, surprise) should be strongly POSITIVE."""
    surprise = torch.linspace(0, 1, 100)
    log_det = -surprise + 0.01 * torch.randn(100)   # planted anti-corr
    uncertainty = (-log_det).tolist()
    rho = spearman_rho(uncertainty, surprise.tolist())
    assert rho > 0.9
    assert rho >= 0.5  # ≥ 0.5 = pass threshold


def test_kill_metric_fails_for_uncorrelated():
    """Random uncorrelated signals should NOT pass the kill criterion."""
    torch.manual_seed(0)
    rho = spearman_rho(torch.randn(200).tolist(), torch.randn(200).tolist())
    assert abs(rho) < 0.5
