"""combined_loss: finite, gradients flow; CE-only when teacher is None or beta=0."""

from __future__ import annotations

import torch

from sprl.training.distill import combined_loss
from sprl.training.teacher_logits import TeacherPack, pack_topk_logits


def _make(N: int = 16, V: int = 64, K: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    s = torch.randn(N, V, requires_grad=True)
    targets = torch.randint(0, V, (N,))
    t_log = torch.randn(N, V) * 1.5
    idx, lg = pack_topk_logits(t_log, top_k=K)
    pack = TeacherPack(topk_indices=idx.long(), topk_logits=lg.float())
    return s, targets, pack


def test_combined_loss_finite_and_grad():
    s, targets, pack = _make()
    loss, parts = combined_loss(s, targets, pack, beta=0.5, T=2.0)
    assert torch.isfinite(loss)
    assert "ce" in parts and "kl" in parts and "total" in parts
    loss.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_combined_loss_pass_through_when_beta_zero():
    s, targets, pack = _make()
    loss, parts = combined_loss(s, targets, pack, beta=0.0, T=2.0)
    # Should equal pure CE.
    expected_ce = torch.nn.functional.cross_entropy(s, targets)
    assert abs(loss.item() - expected_ce.item()) < 1e-6
    assert parts["kl"] == 0.0


def test_combined_loss_pass_through_when_pack_none():
    s, targets, _ = _make()
    loss, parts = combined_loss(s, targets, None, beta=0.5, T=2.0)
    expected_ce = torch.nn.functional.cross_entropy(s, targets)
    assert abs(loss.item() - expected_ce.item()) < 1e-6
    assert parts["kl"] == 0.0


def test_combined_loss_with_leading_dims():
    """Student logits with [B, S, V] should also work."""
    torch.manual_seed(0)
    B, S, V, K = 2, 4, 32, 8
    s = torch.randn(B, S, V, requires_grad=True)
    targets = torch.randint(0, V, (B, S))
    t_log = torch.randn(B, S, V) * 1.0
    idx, lg = pack_topk_logits(t_log, top_k=K)
    pack = TeacherPack(topk_indices=idx.long(), topk_logits=lg.float())
    loss, parts = combined_loss(s, targets, pack, beta=0.3, T=2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(s.grad).all()
