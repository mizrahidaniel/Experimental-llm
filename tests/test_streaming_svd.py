"""Streaming SVD utility: round-trip + recovery of planted top-r spectrum."""

import torch

from sprl.memory.streaming_svd import (
    streaming_svd_topr,
    streaming_svd_topr_aug,
    reconstruction_error,
)


def test_streaming_svd_round_trip():
    torch.manual_seed(0)
    d, r = 16, 4
    U_full = torch.randn(d, r)
    empty = torch.zeros(d, 0)
    U_top = streaming_svd_topr(empty, U_full, r)
    # Reconstruction is exact when the input rank equals the target rank
    # (modulo SVD numerical precision).
    A = U_full @ U_full.t()
    A_top = U_top @ U_top.t()
    assert torch.allclose(A, A_top, atol=1e-4)


def test_streaming_svd_recovers_top_r_planted_spectrum():
    """Plant a strong top-r component and noise; the streaming SVD should
    recover the planted directions with low error."""
    torch.manual_seed(0)
    d, r = 32, 3
    # Strong signal directions × big magnitudes.
    sig_basis, _ = torch.linalg.qr(torch.randn(d, r))
    signal = sig_basis * torch.tensor([10.0, 8.0, 6.0]).unsqueeze(0)
    # Noise.
    noise = torch.randn(d, 5) * 0.1
    U_full = torch.cat([signal, noise], dim=-1)
    empty = torch.zeros(d, 0)
    U_top = streaming_svd_topr(empty, U_full, r)
    err = reconstruction_error(U_full, U_top).item()
    # The relative reconstruction error should be small since signal >> noise.
    base_norm = (U_full @ U_full.t()).flatten().norm().item()
    assert err / base_norm < 0.05


def test_streaming_svd_appending_grows_then_compresses():
    """Append rank-r contributions sequentially; rank stays bounded."""
    torch.manual_seed(0)
    d, r = 12, 4
    U_acc = torch.zeros(d, 0)
    for _ in range(5):
        U_new = torch.randn(d, r) * 0.3
        U_acc = streaming_svd_topr(U_acc, U_new, r)
        assert U_acc.shape == (d, r)


def test_streaming_svd_aug_returns_singular_values():
    torch.manual_seed(1)
    d, r = 8, 3
    U_full = torch.randn(d, 5)
    empty = torch.zeros(d, 0)
    U_top, S_top = streaming_svd_topr_aug(empty, U_full, r)
    assert U_top.shape == (d, r)
    assert S_top.shape == (r,)
    assert (S_top >= 0).all()
    # Singular values descending.
    assert (S_top[:-1] >= S_top[1:]).all()


def test_streaming_svd_batched():
    torch.manual_seed(0)
    B, d, r = 3, 16, 4
    U_acc = torch.zeros(B, d, 0)
    U_new = torch.randn(B, d, r)
    U_top = streaming_svd_topr(U_acc, U_new, r)
    assert U_top.shape == (B, d, r)
