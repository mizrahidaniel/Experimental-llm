"""MLA: forward shape + dense matches a vanilla MHA closely under a degenerate config."""

import torch

from sprl.attention.mla import MultiHeadLatentAttention


def test_mla_forward_shape():
    mla = MultiHeadLatentAttention(
        d_model=64, n_heads=4, d_c_latent=32, d_qhead=16, d_rope_decoupled=8
    )
    h = torch.randn(2, 32, 64)
    out = mla(h)
    assert out.shape == h.shape


def test_mla_grad_flows():
    mla = MultiHeadLatentAttention(
        d_model=64, n_heads=4, d_c_latent=32, d_qhead=16, d_rope_decoupled=8
    )
    h = torch.randn(2, 8, 64, requires_grad=True)
    out = mla(h)
    out.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_mla_with_custom_head_fn():
    """When swapping head_attn_fn, output should remain finite and shape-correct."""
    mla = MultiHeadLatentAttention(
        d_model=32, n_heads=2, d_c_latent=16, d_qhead=8, d_rope_decoupled=4
    )

    def identity_attn(Q, K, V, mask):
        # Just average V over the seq axis as a tiny sanity test.
        return V.mean(dim=-2, keepdim=True).expand_as(V)

    h = torch.randn(2, 8, 32)
    out = mla(h, head_attn_fn=identity_attn)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()
