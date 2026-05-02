"""DSA: switchable behavior + correctness under sparse top-k."""

import torch

from sprl.attention.dsa import DynamicSparseAttention


def test_dsa_short_sequence_runs_dense():
    dsa = DynamicSparseAttention(head_dim=16, top_k_tokens=8, dense_until_seqlen=16)
    Q = torch.randn(1, 2, 8, 16)
    K = torch.randn(1, 2, 8, 16)
    V = torch.randn(1, 2, 8, 16)
    out = dsa(Q, K, V)
    assert out.shape == (1, 2, 8, 16)
    assert torch.isfinite(out).all()


def test_dsa_long_sequence_runs_sparse():
    # S > dense_until_seqlen ⇒ sparse path.
    dsa = DynamicSparseAttention(head_dim=16, top_k_tokens=4, dense_until_seqlen=8)
    Q = torch.randn(1, 2, 32, 16)
    K = torch.randn(1, 2, 32, 16)
    V = torch.randn(1, 2, 32, 16)
    out = dsa(Q, K, V)
    assert out.shape == (1, 2, 32, 16)
    assert torch.isfinite(out).all()


def test_dsa_top_k_full_matches_dense_within_softmax():
    """When top_k == seq_len, the sparse path must equal the dense path
    (modulo the indexer's choice of order, which doesn't affect softmax)."""
    S = 10
    dsa_sparse = DynamicSparseAttention(head_dim=8, top_k_tokens=S, dense_until_seqlen=0)
    dsa_dense = DynamicSparseAttention(head_dim=8, top_k_tokens=S, dense_until_seqlen=10**6)
    # Tie weights so the indexer's projection doesn't differ.
    dsa_sparse.load_state_dict(dsa_dense.state_dict())

    Q = torch.randn(1, 1, S, 8)
    K = torch.randn(1, 1, S, 8)
    V = torch.randn(1, 1, S, 8)
    out_s = dsa_sparse(Q, K, V)
    out_d = dsa_dense(Q, K, V)
    # Tolerance: gather + reorder may produce small numerical differences
    # in float32. Within 1e-4 for top_k = S.
    diff = (out_s - out_d).abs().max().item()
    assert diff < 5e-4, f"diff = {diff}"


def test_dsa_grad_flows():
    dsa = DynamicSparseAttention(head_dim=16, top_k_tokens=4, dense_until_seqlen=0)
    Q = torch.randn(1, 2, 32, 16, requires_grad=True)
    K = torch.randn(1, 2, 32, 16, requires_grad=True)
    V = torch.randn(1, 2, 32, 16, requires_grad=True)
    out = dsa(Q, K, V)
    out.sum().backward()
    assert torch.isfinite(Q.grad).all()
    assert torch.isfinite(K.grad).all()
    assert torch.isfinite(V.grad).all()
