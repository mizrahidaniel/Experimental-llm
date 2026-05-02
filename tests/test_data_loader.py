"""End-to-end test: build_data_iterator from the pilot 200M config."""

from __future__ import annotations

from pathlib import Path

import torch

from sprl.config import SPRLConfig, default_pilot_200m
from sprl.data import DataConfig, build_data_iterator


def _take(it, n):
    out = []
    for i, batch in enumerate(it):
        if i >= n:
            break
        out.append(batch)
    return out


def test_build_iterator_pilot_default_factory():
    cfg = default_pilot_200m()
    cfg.patcher.entropy_threshold = 100.0  # force max-len patches for shape stability
    dc = DataConfig(
        chunk_bytes=128,
        max_seq_patches=8,
        batch_size=2,
        use_hf=False,
        seed=0,
        max_examples_per_source=20,
    )
    it = build_data_iterator(cfg, data_cfg=dc)
    batches = _take(it, 3)
    assert len(batches) == 3
    P = cfg.patcher.max_patch_bytes
    for batch in batches:
        assert batch["patch_emb"].shape == (2, 8, cfg.d_model)
        assert batch["byte_targets"].shape == (2, 8, P)
        assert batch["mtp_targets"].shape == (2, 8, cfg.mtp.depth)
        assert batch["attention_mask"].shape == (2, 8)
        assert batch["lengths"].shape == (2, 8)
        assert batch["patch_bytes"].shape == (2, 8, P)
        assert batch["byte_targets"].dtype == torch.long
        assert batch["patch_emb"].dtype.is_floating_point


def test_build_iterator_with_yaml_config():
    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "pilot_200m_bf16.yaml"
    cfg = SPRLConfig.from_yaml(str(yaml_path))
    cfg.patcher.entropy_threshold = 100.0
    dc = DataConfig(
        chunk_bytes=64,
        max_seq_patches=4,
        batch_size=2,
        use_hf=False,
        seed=0,
        max_examples_per_source=10,
    )
    it = build_data_iterator(cfg, data_cfg=dc)
    batch = next(it)
    assert batch["patch_emb"].shape == (2, 4, cfg.d_model)


def test_build_iterator_subset_domains():
    cfg = default_pilot_200m()
    cfg.patcher.entropy_threshold = 100.0
    dc = DataConfig(
        chunk_bytes=64,
        max_seq_patches=4,
        batch_size=2,
        use_hf=False,
        seed=0,
        domains=["fineweb_edu", "starcoder_v2"],
        max_examples_per_source=10,
    )
    it = build_data_iterator(cfg, data_cfg=dc)
    batches = _take(it, 2)
    assert len(batches) == 2


def test_build_iterator_long_context():
    cfg = default_pilot_200m()
    cfg.patcher.entropy_threshold = 100.0
    dc = DataConfig(
        chunk_bytes=2048,
        max_seq_patches=128,
        batch_size=1,
        use_hf=False,
        seed=0,
        long_context_enabled=True,
        long_context_seq_bytes=4096,
        domains=["long_context"],
        max_examples_per_source=4,
    )
    it = build_data_iterator(cfg, data_cfg=dc)
    batch = next(it)
    assert batch["patch_emb"].shape[0] == 1
    assert batch["patch_emb"].shape[2] == cfg.d_model


def test_build_iterator_determinism():
    cfg = default_pilot_200m()
    cfg.patcher.entropy_threshold = 100.0
    dc = DataConfig(
        chunk_bytes=64,
        max_seq_patches=4,
        batch_size=2,
        use_hf=False,
        seed=123,
        max_examples_per_source=20,
    )
    # Build two iterators with the same data_cfg seed; the *byte stream*
    # interleaving must be identical. (The byte_encoder weights differ between
    # builds, so we compare patch_bytes — the deterministic upstream output.)
    it1 = build_data_iterator(cfg, data_cfg=dc)
    it2 = build_data_iterator(cfg, data_cfg=dc)
    b1 = next(it1)
    b2 = next(it2)
    assert torch.equal(b1["patch_bytes"], b2["patch_bytes"])
    assert torch.equal(b1["attention_mask"], b2["attention_mask"])


def test_build_iterator_curriculum():
    cfg = default_pilot_200m()
    cfg.patcher.entropy_threshold = 100.0
    cfg.training.curriculum_freeze_until_tokens = 0
    dc = DataConfig(
        chunk_bytes=64,
        max_seq_patches=4,
        batch_size=2,
        use_hf=False,
        seed=0,
        use_curriculum=True,
        max_examples_per_source=10,
    )
    it = build_data_iterator(cfg, data_cfg=dc)
    batch = next(it)
    assert batch["patch_emb"].shape == (2, 4, cfg.d_model)
