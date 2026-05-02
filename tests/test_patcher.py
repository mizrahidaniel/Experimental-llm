"""Patcher invariants — average patch size, byte_lm forward."""

import torch

from sprl.patcher import ByteLM, EntropyPatcher, patch_bytes_with_entropy


def test_byte_lm_forward_shape():
    lm = ByteLM(dim=64, n_layers=2, n_heads=4)
    bytes_in = torch.randint(0, 256, (2, 32))
    logits = lm(bytes_in)
    assert logits.shape == (2, 32, 256)


def test_byte_lm_entropy_in_range():
    lm = ByteLM(dim=64, n_layers=2)
    bytes_in = torch.randint(0, 256, (2, 32))
    H = lm.next_byte_entropy(bytes_in)
    assert H.shape == (2, 32)
    # Max entropy for 256 classes is log(256) ≈ 5.545.
    assert (H <= 5.55 + 1e-3).all()
    assert (H >= 0).all()


def test_entropy_patcher_min_max_bounds():
    bytes_seq = torch.randint(0, 256, (200,))
    H = torch.rand(200) * 3.0  # random entropies in [0, 3]
    plan = patch_bytes_with_entropy(
        bytes_seq, H, threshold=1.5, min_patch_bytes=1, max_patch_bytes=8
    )
    assert plan.n_patches > 0
    for s, e in zip(plan.starts, plan.ends):
        assert 1 <= e - s <= 8
    # Patches cover the whole stream.
    assert plan.starts[0] == 0
    assert plan.ends[-1] == 200


def test_entropy_patcher_low_entropy_makes_long_patches():
    bytes_seq = torch.randint(0, 256, (256,))
    # All entropies far below threshold ⇒ patches should saturate to max_patch_bytes.
    H = torch.zeros(256)
    plan = patch_bytes_with_entropy(
        bytes_seq, H, threshold=1.5, min_patch_bytes=1, max_patch_bytes=16
    )
    # All patches should be max length except possibly the last.
    sizes = [e - s for s, e in zip(plan.starts, plan.ends)]
    for sz in sizes[:-1]:
        assert sz == 16


def test_high_entropy_makes_short_patches():
    bytes_seq = torch.randint(0, 256, (32,))
    H = torch.full((32,), 5.0)  # always above threshold
    plan = patch_bytes_with_entropy(
        bytes_seq, H, threshold=1.5, min_patch_bytes=1, max_patch_bytes=16
    )
    sizes = [e - s for s, e in zip(plan.starts, plan.ends)]
    # With min=1, every position closes the patch ⇒ many size-1 patches.
    assert max(sizes) <= 2  # min satisfied + immediate close


def test_entropy_patcher_module_runs():
    lm = ByteLM(dim=32, n_layers=1)
    p = EntropyPatcher(lm, threshold=1.0, min_patch_bytes=1, max_patch_bytes=8)
    bytes_in = torch.randint(0, 256, (2, 64))
    plans = p.plan(bytes_in)
    assert len(plans) == 2
    for plan in plans:
        assert plan.starts[0] == 0
        assert plan.ends[-1] == 64
