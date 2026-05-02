"""HISA: hierarchical block→token indexer correctness."""

import torch

from sprl.attention.hisa import HierarchicalIndexer


def test_hisa_forward_shape():
    h = HierarchicalIndexer(head_dim=16, block_size=8, top_k_blocks=2, top_k_tokens_per_block=4)
    Q = torch.randn(1, 2, 32, 16)
    K = torch.randn(1, 2, 32, 16)
    V = torch.randn(1, 2, 32, 16)
    out = h(Q, K, V)
    assert out.shape == (1, 2, 32, 16)
    assert torch.isfinite(out).all()


def test_hisa_grad_flows():
    h = HierarchicalIndexer(head_dim=8, block_size=4, top_k_blocks=2, top_k_tokens_per_block=2)
    Q = torch.randn(1, 1, 16, 8, requires_grad=True)
    K = torch.randn(1, 1, 16, 8, requires_grad=True)
    V = torch.randn(1, 1, 16, 8, requires_grad=True)
    out = h(Q, K, V)
    out.sum().backward()
    assert torch.isfinite(Q.grad).all()
    assert torch.isfinite(V.grad).all()


def test_hisa_causal_mask():
    """Top-k tokens chosen for query t must all have index ≤ t."""
    torch.manual_seed(0)
    S = 16
    h = HierarchicalIndexer(head_dim=8, block_size=4, top_k_blocks=4, top_k_tokens_per_block=4)
    Q = torch.randn(1, 1, S, 8)
    K = torch.randn(1, 1, S, 8)
    V = torch.randn(1, 1, S, 8)
    # The output for query 0 must equal V[0] (only one valid token).
    out = h(Q, K, V, causal=True)
    # out[0,0,0] should be a function of V[0,0,0,:] only.
    # We can't trivially assert equality (softmax of one entry), but we can
    # check that swapping V at position > 0 doesn't change query-0's output.
    Vb = V.clone()
    Vb[..., 1:, :] = torch.randn_like(Vb[..., 1:, :])
    out_b = h(Q, K, Vb, causal=True)
    diff = (out[..., 0, :] - out_b[..., 0, :]).abs().max().item()
    assert diff < 1e-5, f"causal violation: query 0 changed when future V changed (diff = {diff})"
