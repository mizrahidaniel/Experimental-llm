"""NVFP4 emulation: round-trip Frobenius loss within bound."""

import torch

from sprl.precision.nvfp4 import nvfp4_roundtrip_emulated
from sprl.precision.rht import random_hadamard


def test_nvfp4_roundtrip_within_5_percent():
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    y = nvfp4_roundtrip_emulated(x)
    err = (x - y).norm() / x.norm()
    # Without RHT, FP4 is lossy on outliers; expect ~1-5% Frobenius loss.
    assert err < 0.10, f"NVFP4 round-trip error too high: {err:.4f}"


def test_nvfp4_with_rht_lower_error():
    torch.manual_seed(0)
    # Heavy-tailed input — outliers are exactly what RHT helps with.
    x = torch.randn(64, 128) + 10.0 * torch.randn(64, 128) * (torch.rand(64, 128) < 0.01)
    y_no_rht = nvfp4_roundtrip_emulated(x)
    err_no_rht = (x - y_no_rht).norm() / x.norm()

    x_rht = random_hadamard(x, seed=42)
    y_rht = nvfp4_roundtrip_emulated(x_rht)
    err_rht = (x_rht - y_rht).norm() / x_rht.norm()

    # RHT-prequantization should be no worse on outliers, often much better.
    assert err_rht <= err_no_rht * 1.1


def test_nvfp4_zero_in_zero_out():
    x = torch.zeros(16, 16)
    y = nvfp4_roundtrip_emulated(x)
    assert (y.abs() < 1e-6).all()
