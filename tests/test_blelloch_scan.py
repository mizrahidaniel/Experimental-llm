"""Blelloch scan: equivalence to sequential scan + finite-output invariants."""

import torch

from sprl.memory.parallel_scan import (
    associative_scan,
    blelloch_scan,
    chunked_associative_scan,
)


def _add(a, b):
    return (a[0] + b[0],)


def test_blelloch_matches_sequential_addition():
    x = torch.arange(13, dtype=torch.float32).reshape(1, 13, 1)
    seq = associative_scan((x,), _add, axis=1)
    bl = blelloch_scan((x,), _add, axis=1)
    assert torch.allclose(seq[0], bl[0])


def test_blelloch_matches_sequential_random_lengths():
    torch.manual_seed(0)
    for T in [1, 2, 3, 4, 5, 7, 8, 16, 17, 31, 32, 33]:
        x = torch.randn(2, T, 4)
        seq = associative_scan((x,), _add, axis=1)
        bl = blelloch_scan((x,), _add, axis=1)
        assert torch.allclose(seq[0], bl[0], atol=1e-5), f"failed at T={T}"


def test_blelloch_matrix_multiply_associative_non_commutative():
    """A non-trivial associative non-commutative combine: 4x4 matrix multiply."""
    torch.manual_seed(0)
    B, T, d = 2, 17, 4
    M = torch.randn(B, T, d, d) * 0.2 + torch.eye(d).expand(B, T, d, d)

    def matmul(a, b):
        # b after a: combine(a, b) = b @ a (left-to-right scan composition).
        return (b[0] @ a[0],)

    seq = associative_scan((M,), matmul, axis=1)
    bl = blelloch_scan((M,), matmul, axis=1)
    assert torch.allclose(seq[0], bl[0], atol=1e-4)


def test_blelloch_no_nan_under_chunked_path():
    """The chunked associative scan and Blelloch scan should both yield
    finite outputs on a moderately long sequence."""
    torch.manual_seed(0)
    x = torch.randn(1, 200, 8)
    chunked = chunked_associative_scan((x,), _add, chunk_size=32, axis=1)
    bl = blelloch_scan((x,), _add, axis=1)
    seq = associative_scan((x,), _add, axis=1)
    assert torch.isfinite(chunked[0]).all()
    assert torch.isfinite(bl[0]).all()
    assert torch.allclose(chunked[0], seq[0], atol=1e-4)
    assert torch.allclose(bl[0], seq[0], atol=1e-4)


def test_blelloch_grad_flows():
    """Backprop through the Blelloch scan must work."""
    x = torch.randn(1, 8, 4, requires_grad=True)
    out = blelloch_scan((x,), _add, axis=1)
    out[0].sum().backward()
    assert torch.isfinite(x.grad).all()


def test_blelloch_single_element():
    x = torch.randn(1, 1, 3)
    bl = blelloch_scan((x,), _add, axis=1)
    assert torch.allclose(bl[0], x)


def test_blelloch_multi_tensor_pytree():
    """Combine two tensors at once (e.g. (running_sum, running_count))."""
    x = torch.randn(1, 8, 3)
    counts = torch.ones(1, 8, 1)

    def combine(a, b):
        return (a[0] + b[0], a[1] + b[1])

    seq = associative_scan((x, counts), combine, axis=1)
    bl = blelloch_scan((x, counts), combine, axis=1)
    assert torch.allclose(seq[0], bl[0], atol=1e-5)
    assert torch.allclose(seq[1], bl[1], atol=1e-5)
