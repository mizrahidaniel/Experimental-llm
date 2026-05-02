"""TTT-mode streaming Kalman: matches chunked-scan output."""

import torch

from sprl.memory.kalman_info import KalmanInfoMemory
from sprl.memory.ttt_mode import TestTimeKalmanState, surprise_residual, update


def test_streaming_update_matches_chunked_eta_D():
    """Streaming update over a sequence should match the chunked scan output
    on (η, D) within tolerance (U may differ due to per-step compression)."""
    torch.manual_seed(0)
    km = KalmanInfoMemory(d_model=8, rank=2, chunk_size=16, strict_rank=True)
    h = torch.randn(1, 10, 8)
    out_chunked = km(h)

    state = TestTimeKalmanState()
    eta_stream = []
    D_stream = []
    for t in range(h.shape[1]):
        eta, D, U = update(km, state, h[:, t])
        eta_stream.append(eta)
        D_stream.append(D)
    eta_stream = torch.stack(eta_stream, dim=1)
    D_stream = torch.stack(D_stream, dim=1)

    assert torch.allclose(eta_stream, out_chunked["eta"], atol=1e-3)
    assert torch.allclose(D_stream, out_chunked["D"], atol=1e-3)


def test_ttt_state_increments_n_steps():
    km = KalmanInfoMemory(d_model=8, rank=2, strict_rank=True)
    state = TestTimeKalmanState()
    h = torch.randn(1, 8)
    for _ in range(3):
        update(km, state, h)
    assert state.n_steps == 3


def test_ttt_state_reset_clears_state():
    km = KalmanInfoMemory(d_model=8, rank=2, strict_rank=True)
    state = TestTimeKalmanState()
    update(km, state, torch.randn(1, 8))
    assert state.eta is not None
    state.reset()
    assert state.eta is None
    assert state.n_steps == 0


def test_surprise_residual_zero_on_empty_state():
    km = KalmanInfoMemory(d_model=8, rank=2, strict_rank=True)
    state = TestTimeKalmanState()
    h = torch.randn(1, 8)
    r = surprise_residual(km, state, h)
    assert torch.allclose(r, torch.zeros_like(r))


def test_surprise_residual_finite_after_updates():
    km = KalmanInfoMemory(d_model=8, rank=2, strict_rank=True)
    state = TestTimeKalmanState()
    for _ in range(5):
        update(km, state, torch.randn(1, 8))
    r = surprise_residual(km, state, torch.randn(1, 8), lr=1e-2)
    assert torch.isfinite(r).all()


def test_ttt_legacy_mode_also_matches_chunked():
    """With strict_rank=False the streaming path should *exactly* match
    the chunked path on (η, D, U) — no SVD compression involved."""
    torch.manual_seed(0)
    km = KalmanInfoMemory(d_model=8, rank=2, chunk_size=16, strict_rank=False)
    h = torch.randn(1, 7, 8)
    out_chunked = km(h)

    state = TestTimeKalmanState()
    etas, Ds, Us = [], [], []
    for t in range(h.shape[1]):
        eta, D, U = update(km, state, h[:, t])
        etas.append(eta)
        Ds.append(D)
        Us.append(U)
    eta_stream = torch.stack(etas, dim=1)
    D_stream = torch.stack(Ds, dim=1)
    U_stream = torch.stack(Us, dim=1)

    assert torch.allclose(eta_stream, out_chunked["eta"], atol=1e-4)
    assert torch.allclose(D_stream, out_chunked["D"], atol=1e-4)
    assert torch.allclose(U_stream, out_chunked["U"], atol=1e-4)
