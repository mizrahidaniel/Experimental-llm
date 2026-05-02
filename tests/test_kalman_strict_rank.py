"""Kalman info-form with strict_rank=True: log_det stability + grad flow."""

import torch

from sprl.memory.kalman_info import KalmanInfoMemory


def test_strict_rank_log_det_finite_long_sequence():
    """log det Λ_t must stay finite over a long sequence."""
    torch.manual_seed(0)
    km = KalmanInfoMemory(d_model=16, rank=4, chunk_size=8, strict_rank=True)
    h = torch.randn(2, 64, 16)
    out = km(h)
    assert torch.isfinite(out["log_det_Lambda"]).all()
    assert out["U"].shape == (2, 64, 16, 4)  # rank fixed at 4


def test_strict_rank_grad_flows():
    km = KalmanInfoMemory(d_model=8, rank=2, chunk_size=4, strict_rank=True)
    h = torch.randn(1, 12, 8, requires_grad=True)
    out = km(h)
    loss = out["eta"].pow(2).mean() + out["log_det_Lambda"].mean()
    loss.backward()
    assert torch.isfinite(h.grad).all()


def test_strict_rank_fixed_rank_invariant():
    """At every position, U has shape [..., d, rank]."""
    km = KalmanInfoMemory(d_model=12, rank=3, chunk_size=5, strict_rank=True)
    h = torch.randn(2, 30, 12)
    out = km(h)
    assert out["U"].shape == (2, 30, 12, 3)


def test_strict_rank_streaming_step_matches_chunked_short():
    """For a short sequence, streaming_step should approximate the chunked
    output of the same Kalman module."""
    torch.manual_seed(0)
    km = KalmanInfoMemory(d_model=8, rank=2, chunk_size=16, strict_rank=True)
    h = torch.randn(1, 6, 8)
    out_chunked = km(h)

    # Streaming.
    state = None
    etas, Ds, Us = [], [], []
    for t in range(h.shape[1]):
        eta, D, U = km.streaming_step(h[:, t], state)
        state = (eta, D, U)
        etas.append(eta)
        Ds.append(D)
        Us.append(U)
    eta_stream = torch.stack(etas, dim=1)
    D_stream = torch.stack(Ds, dim=1)

    # eta and D should match closely (no SVD on these); U may differ due to
    # SVD-based per-step compression vs end-of-chunk compression — we only
    # check eta and D here.
    assert torch.allclose(eta_stream, out_chunked["eta"], atol=1e-3)
    assert torch.allclose(D_stream, out_chunked["D"], atol=1e-3)


def test_strict_vs_legacy_both_finite_and_well_conditioned():
    torch.manual_seed(0)
    km_strict = KalmanInfoMemory(d_model=8, rank=2, strict_rank=True, chunk_size=4)
    km_legacy = KalmanInfoMemory(d_model=8, rank=2, strict_rank=False, chunk_size=4)
    # Match weights so the only difference is the combine.
    km_legacy.load_state_dict(km_strict.state_dict())
    h = torch.randn(1, 20, 8)
    o1 = km_strict(h)
    o2 = km_legacy(h)
    assert torch.isfinite(o1["log_det_Lambda"]).all()
    assert torch.isfinite(o2["log_det_Lambda"]).all()
    # eta and D paths are identical between strict and legacy (only U differs).
    assert torch.allclose(o1["eta"], o2["eta"], atol=1e-5)
    assert torch.allclose(o1["D"], o2["D"], atol=1e-5)
