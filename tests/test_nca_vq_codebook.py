"""VQ codebook: deterministic assignment, EMA convergence, round-trip."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sprl.training.nca_pretrain import (
    GrowingNCA,
    VQCodebook,
    tokenize_2x2_patches_lsh,
    tokenize_2x2_patches_vq,
)


def test_assign_deterministic_for_fixed_input():
    torch.manual_seed(0)
    cb = VQCodebook(vocab_size=64, code_dim=8, init_scale=0.3)
    x = torch.randn(32, 8)
    a = cb.assign(x)
    b = cb.assign(x)
    assert torch.equal(a, b)


def test_ema_converges_on_synthetic_clusters():
    """Two well-separated cluster blobs should be picked up by EMA."""
    torch.manual_seed(0)
    D = 8
    K = 8
    cb = VQCodebook(vocab_size=K, code_dim=D, decay=0.5, init_scale=0.05)
    cb.train()

    # Two separated cluster centers on the unit sphere.
    c1 = F.normalize(torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dim=0)
    c2 = F.normalize(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0]), dim=0)

    for step in range(100):
        n = 64
        choose = (torch.rand(n) > 0.5).float().unsqueeze(-1)
        x = choose * c1 + (1 - choose) * c2
        x = x + 0.01 * torch.randn_like(x)
        cb.quantize(x)

    cb.eval()
    # After training the codebook should have at least one code very close to each centroid.
    sim_c1 = (cb.codes @ c1).max().item()
    sim_c2 = (cb.codes @ c2).max().item()
    assert sim_c1 > 0.95, f"cluster 1 not captured (sim={sim_c1})"
    assert sim_c2 > 0.95, f"cluster 2 not captured (sim={sim_c2})"


def test_quantize_straight_through_gradient():
    torch.manual_seed(0)
    cb = VQCodebook(vocab_size=32, code_dim=8, init_scale=0.1)
    x = torch.randn(16, 8, requires_grad=True)
    z_q, idx, commit = cb.quantize(x)
    loss = z_q.sum() + commit
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_tokenize_2x2_patches_vq_round_trip_consistent():
    torch.manual_seed(0)
    nca = GrowingNCA(n_channels=16, hidden=32)
    nca.eval()
    x = torch.zeros(1, 16, 8, 8)
    x[:, :, 4, 4] = 1.0
    with torch.no_grad():
        traj = nca(x, n_steps=4)
    state = traj[-1][0]  # [C, H, W]

    cb = VQCodebook(vocab_size=128, code_dim=4 * 16, init_scale=0.3)

    a = tokenize_2x2_patches_vq(state, cb)
    b = tokenize_2x2_patches_vq(state, cb)
    assert torch.equal(a, b)
    # Number of tokens equals number of 2x2 patches.
    H, W = state.shape[1], state.shape[2]
    assert a.shape == ((H // 2) * (W // 2),)


def test_lsh_fallback_still_works():
    torch.manual_seed(0)
    # 4 channels -> 16-bit signature, no overflow.
    state = torch.randn(4, 8, 8)
    codes = tokenize_2x2_patches_lsh(state, vocab_size=512)
    assert codes.shape == (16,)
    assert (codes >= 0).all() and (codes < 512).all()


def test_codebook_dim_mismatch_raises():
    cb = VQCodebook(vocab_size=16, code_dim=32)
    state = torch.randn(16, 8, 8)  # patch dim = 4*16 = 64, != 32
    try:
        tokenize_2x2_patches_vq(state, cb)
        assert False, "expected ValueError"
    except ValueError:
        pass
