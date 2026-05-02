"""Speculative decoding: produces n_new bytes + tracks acceptance rate."""

import torch

from sprl.config import SPRLConfig
from sprl.inference.speculative import speculative_decode
from sprl.model import SPRLv2


def _tiny_cfg() -> SPRLConfig:
    cfg = SPRLConfig(d_model=64, n_layers=2, max_seq_len_patches=64)
    cfg.attention.mla.n_heads = 4
    cfg.attention.mla.d_c_latent = 32
    cfg.attention.mla.d_qhead = 16
    cfg.attention.mla.d_rope_decoupled = 8
    cfg.attention.dsa.dense_until_seqlen = 256
    cfg.attention.sliding.window_size = 32
    cfg.moe.num_routed_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 32
    cfg.patcher.byte_encoder_dim = 32
    cfg.patcher.max_patch_bytes = 4
    cfg.mtp.depth = 2
    return cfg


def test_speculative_decode_returns_n_new():
    torch.manual_seed(0)
    m = SPRLv2(_tiny_cfg())
    g = torch.Generator().manual_seed(0)
    out, rate = speculative_decode(
        m, b"hi", n_new=5, mtp_depth=2, temperature=1.0, generator=g
    )
    assert isinstance(out, bytes)
    assert len(out) >= 5  # may slightly overrun on accept
    assert 0.0 <= rate <= 1.0


def test_speculative_decode_deterministic_with_generator():
    torch.manual_seed(0)
    m = SPRLv2(_tiny_cfg())
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    out1, _ = speculative_decode(m, b"hi", n_new=4, mtp_depth=2, generator=g1)
    out2, _ = speculative_decode(m, b"hi", n_new=4, mtp_depth=2, generator=g2)
    assert out1 == out2


def test_speculative_decode_requires_mtp():
    """Disabling MTP should raise."""
    cfg = _tiny_cfg()
    cfg.mtp.enabled = False
    m = SPRLv2(cfg)
    import pytest

    with pytest.raises(RuntimeError):
        speculative_decode(m, b"hi", n_new=2)
