"""Sparse top-K KL: matches dense KL when K=V, and gradients flow."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sprl.training.teacher_logits import pack_topk_logits, sparse_top_k_kl_loss


def _dense_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T ** 2)


def test_sparse_matches_dense_at_full_k():
    torch.manual_seed(0)
    N, V = 8, 32
    s_log = torch.randn(N, V) * 1.5
    t_log = torch.randn(N, V) * 1.5
    T = 2.0

    idx, lg = pack_topk_logits(t_log, top_k=V)
    sparse = sparse_top_k_kl_loss(s_log, idx.long(), lg.float(), T=T)
    dense = _dense_kl(s_log, t_log, T=T)
    assert torch.allclose(sparse, dense, atol=1e-4), f"sparse={sparse.item()}, dense={dense.item()}"


def test_sparse_gradient_flows():
    torch.manual_seed(1)
    N, V, K = 4, 16, 8
    s_log = torch.randn(N, V, requires_grad=True)
    t_log = torch.randn(N, V)
    idx, lg = pack_topk_logits(t_log, top_k=K)
    loss = sparse_top_k_kl_loss(s_log, idx.long(), lg.float(), T=2.0)
    loss.backward()
    assert s_log.grad is not None
    assert torch.isfinite(s_log.grad).all()
    assert s_log.grad.abs().sum().item() > 0


def test_sparse_gradient_gradcheck():
    torch.manual_seed(2)
    N, V, K = 2, 8, 4

    t_log = torch.randn(N, V, dtype=torch.double)
    idx, lg = pack_topk_logits(t_log, top_k=K)
    idx_l = idx.long()
    lg_d = lg.to(torch.float64)

    def f(s_log: torch.Tensor) -> torch.Tensor:
        return sparse_top_k_kl_loss(s_log, idx_l, lg_d, T=2.0)

    s_log = torch.randn(N, V, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(f, (s_log,), eps=1e-6, atol=1e-4)


def test_sparse_with_partial_k_close_when_teacher_peaked():
    """Peaked teacher: sparse-K with K < V should still ~ dense KL."""
    torch.manual_seed(3)
    N, V, K = 4, 64, 16
    t_log = torch.zeros(N, V)
    t_log[:, 0] = 30.0
    s_log = torch.randn(N, V) * 1.5
    idx, lg = pack_topk_logits(t_log, top_k=K)
    sparse = sparse_top_k_kl_loss(s_log, idx.long(), lg.float(), T=2.0)
    dense = _dense_kl(s_log, t_log, T=2.0)
    assert torch.allclose(sparse, dense, atol=5e-3), f"sparse={sparse.item()}, dense={dense.item()}"
