"""Tropical attention numerical stability + soft-tropical → tropical limit."""

import torch

from sprl.attention.tropical import (
    TropicalAttentionHead,
    soft_tropical_attn_forward,
    tropical_attn_forward,
    tropical_inner,
)


def test_tropical_inner_correct():
    Q = torch.tensor([[[1.0, 2.0, 0.5]]])  # [1, 1, 3]
    K = torch.tensor([[[0.0, 1.0, 1.0], [3.0, 0.0, 0.0]]])  # [1, 2, 3]
    s = tropical_inner(Q, K)
    # s[0,0] = max(1+0, 2+1, 0.5+1) = 3
    # s[0,1] = max(1+3, 2+0, 0.5+0) = 4
    assert s.shape == (1, 1, 2)
    assert torch.allclose(s[0, 0, 0], torch.tensor(3.0))
    assert torch.allclose(s[0, 0, 1], torch.tensor(4.0))


def test_soft_tropical_converges_to_tropical():
    Q = torch.randn(1, 1, 8, 4)
    K = torch.randn(1, 1, 8, 4)
    V = torch.randn(1, 1, 8, 4)
    out_hard = tropical_attn_forward(Q, K, V)
    out_soft_high = soft_tropical_attn_forward(Q, K, V, beta=64.0)
    diff = (out_hard - out_soft_high).abs().max().item()
    assert diff < 0.5, f"soft (β=64) should be close to hard tropical, got {diff}"


def test_dual_algebra_head_forward():
    head = TropicalAttentionHead(n_heads=4, fraction_tropical=0.25, beta_init=8.0)
    Q = torch.randn(1, 4, 16, 8)
    K = torch.randn(1, 4, 16, 8)
    V = torch.randn(1, 4, 16, 8)
    out = head(Q, K, V)
    assert out.shape == V.shape
    assert torch.isfinite(out).all()


def test_dual_algebra_head_grad():
    head = TropicalAttentionHead(n_heads=4, fraction_tropical=0.5, beta_init=8.0)
    Q = torch.randn(1, 4, 16, 8, requires_grad=True)
    K = torch.randn(1, 4, 16, 8, requires_grad=True)
    V = torch.randn(1, 4, 16, 8, requires_grad=True)
    out = head(Q, K, V)
    out.sum().backward()
    assert torch.isfinite(Q.grad).all()
    assert torch.isfinite(K.grad).all()
    assert torch.isfinite(V.grad).all()


def test_tropical_high_beta_stability():
    """At β = 64, the softmax surrogate should remain finite (no overflow)."""
    Q = 5.0 * torch.randn(1, 1, 8, 4)
    K = 5.0 * torch.randn(1, 1, 8, 4)
    V = 5.0 * torch.randn(1, 1, 8, 4)
    out = soft_tropical_attn_forward(Q, K, V, beta=64.0)
    assert torch.isfinite(out).all()
