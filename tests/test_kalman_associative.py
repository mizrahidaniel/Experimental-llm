"""Kalman info-form: associativity of the scan, and forward correctness."""

import torch

from sprl.memory.kalman_info import KalmanInfoMemory
from sprl.memory.parallel_scan import (
    associative_scan,
    chunked_associative_scan,
)


def test_associative_scan_matches_sequential_for_addition():
    elements = (torch.arange(10, dtype=torch.float32).unsqueeze(0),)  # [1, 10]
    out_seq = associative_scan(elements, lambda a, b: (a[0] + b[0],), axis=1)
    # Cumulative sum.
    expected = torch.cumsum(elements[0], dim=1)
    assert torch.allclose(out_seq[0], expected)


def test_chunked_scan_matches_sequential():
    x = torch.randn(1, 32, 4)  # [B, T, d]
    elements = (x,)
    seq = associative_scan(elements, lambda a, b: (a[0] + b[0],), axis=1)
    chunked = chunked_associative_scan(
        elements, lambda a, b: (a[0] + b[0],), chunk_size=8, axis=1
    )
    assert torch.allclose(seq[0], chunked[0], atol=1e-5)


def test_kalman_combine_associative():
    """The combine function must be associative: f(f(a,b), c) == f(a, f(b,c))."""
    torch.manual_seed(0)
    d, r = 4, 2
    F_a = torch.rand(d) * 0.9 + 0.05
    F_b = torch.rand(d) * 0.9 + 0.05
    F_c = torch.rand(d) * 0.9 + 0.05
    eta_a, eta_b, eta_c = torch.randn(d), torch.randn(d), torch.randn(d)
    D_a, D_b, D_c = torch.rand(d) + 0.1, torch.rand(d) + 0.1, torch.rand(d) + 0.1
    U_a, U_b, U_c = torch.randn(d, r), torch.randn(d, r), torch.randn(d, r)

    a = (F_a, eta_a, D_a, U_a)
    b = (F_b, eta_b, D_b, U_b)
    c = (F_c, eta_c, D_c, U_c)
    f = KalmanInfoMemory._combine
    left = f(f(a, b), c)
    right = f(a, f(b, c))
    # F, eta, D associative exactly.
    for li, ri in zip(left[:3], right[:3]):
        assert torch.allclose(li, ri, atol=1e-6), "non-associative on F/eta/D"
    # U under the rank-truncated combine: U_c = U_b + F_b · U_a is associative.
    assert torch.allclose(left[3], right[3], atol=1e-5)


def test_kalman_forward_shape_and_log_det():
    km = KalmanInfoMemory(d_model=16, rank=4, chunk_size=8)
    h = torch.randn(2, 16, 16)
    out = km(h)
    assert out["eta"].shape == (2, 16, 16)
    assert out["D"].shape == (2, 16, 16)
    assert out["log_det_Lambda"].shape == (2, 16)
    assert torch.isfinite(out["log_det_Lambda"]).all()


def test_kalman_grad_flows():
    km = KalmanInfoMemory(d_model=8, rank=2, chunk_size=4)
    h = torch.randn(1, 8, 8, requires_grad=True)
    out = km(h)
    loss = out["eta"].pow(2).mean() + out["log_det_Lambda"].mean()
    loss.backward()
    assert torch.isfinite(h.grad).all()
