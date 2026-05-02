"""Diagnostic helpers used by the kill-criterion harness."""

import math

import numpy as np
import torch

from sprl.utils.diagnostics import (
    argmax_concentration,
    log_det_lowrank,
    operator_distance_from_identity,
    power_law_fit,
    spearman_rho,
)


def test_spearman_rho_perfect():
    x = list(range(20))
    y = list(range(20))
    assert abs(spearman_rho(x, y) - 1.0) < 1e-6


def test_spearman_rho_anticorrelated():
    x = list(range(20))
    y = list(range(20, 0, -1))
    assert abs(spearman_rho(x, y) + 1.0) < 1e-6


def test_spearman_rho_constant():
    x = [1.0] * 10
    y = list(range(10))
    assert spearman_rho(x, y) == 0.0


def test_power_law_fit_recovers_exponent():
    iters = list(range(1, 50))
    losses = [10.0 / (i ** 0.5) for i in iters]
    a, b, r2 = power_law_fit(iters, losses)
    assert abs(b - 0.5) < 1e-3
    assert r2 > 0.99


def test_operator_distance_zero_for_identity():
    eye = torch.eye(8)
    assert operator_distance_from_identity(eye) < 1e-6


def test_operator_distance_nonzero_for_random():
    M = torch.randn(8, 8)
    assert operator_distance_from_identity(M) > 0.5


def test_argmax_concentration_full_when_identical():
    a = torch.eye(4).unsqueeze(0)
    assert argmax_concentration(a, a) == 1.0


def test_log_det_lowrank_matches_full():
    torch.manual_seed(0)
    D = torch.rand(8) + 0.1
    U = torch.randn(8, 2)
    via = log_det_lowrank(D, U)
    full = torch.logdet(torch.diag(D) + U @ U.t())
    assert torch.allclose(via, full, atol=1e-4)
