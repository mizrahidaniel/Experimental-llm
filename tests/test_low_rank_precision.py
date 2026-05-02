"""SMW + matrix-determinant-lemma correctness."""

import torch

from sprl.memory.low_rank_precision import (
    log_det_low_rank,
    smw_inverse_apply,
)


def test_smw_inverse_matches_explicit():
    torch.manual_seed(0)
    d, r = 8, 3
    D = torch.rand(d) + 0.1
    U = torch.randn(d, r)
    A = torch.diag(D) + U @ U.t()
    x = torch.randn(d)

    explicit = torch.linalg.solve(A, x)
    via_smw = smw_inverse_apply(D, U, x)
    assert torch.allclose(explicit, via_smw, atol=1e-4)


def test_smw_inverse_batched():
    B = 4
    D = torch.rand(B, 8) + 0.1
    U = torch.randn(B, 8, 3)
    x = torch.randn(B, 8)
    via_smw = smw_inverse_apply(D, U, x)
    for b in range(B):
        A = torch.diag(D[b]) + U[b] @ U[b].t()
        explicit = torch.linalg.solve(A, x[b])
        assert torch.allclose(explicit, via_smw[b], atol=1e-4)


def test_log_det_low_rank_correct():
    torch.manual_seed(0)
    d, r = 6, 2
    D = torch.rand(d) + 0.1
    U = torch.randn(d, r)
    A = torch.diag(D) + U @ U.t()
    expected = torch.logdet(A)
    via = log_det_low_rank(D, U)
    assert torch.allclose(expected, via, atol=1e-4)


def test_log_det_grad_flows():
    D = (torch.rand(8) + 0.1).requires_grad_(True)
    U = torch.randn(8, 2, requires_grad=True)
    out = log_det_low_rank(D, U)
    out.backward()
    assert torch.isfinite(D.grad).all()
    assert torch.isfinite(U.grad).all()
