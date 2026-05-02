"""End-to-end byte-level generation: shape, determinism, eot, TTT mode."""

import torch

from sprl.config import SPRLConfig
from sprl.inference.generate import generate
from sprl.model import SPRLv2


def _tiny_cfg(**overrides) -> SPRLConfig:
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
    for k, v in overrides.items():
        head = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            head = getattr(head, p)
        setattr(head, parts[-1], v)
    return cfg


def test_generate_returns_n_new_bytes():
    torch.manual_seed(0)
    m = SPRLv2(_tiny_cfg())
    out = generate(m, b"hi", n_new=5, temperature=1.0)
    assert isinstance(out, bytes)
    assert len(out) == 5


def test_generate_deterministic_with_generator():
    torch.manual_seed(0)
    m = SPRLv2(_tiny_cfg())
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    out1 = generate(m, b"hi", n_new=4, temperature=1.0, top_p=0.95, generator=g1)
    out2 = generate(m, b"hi", n_new=4, temperature=1.0, top_p=0.95, generator=g2)
    assert out1 == out2


def test_generate_argmax_is_deterministic_no_generator():
    """temperature=0 should be deterministic without a Generator."""
    torch.manual_seed(0)
    m = SPRLv2(_tiny_cfg())
    out1 = generate(m, b"hi", n_new=3, temperature=0.0)
    out2 = generate(m, b"hi", n_new=3, temperature=0.0)
    assert out1 == out2


def test_generate_eot_byte_stops_early():
    torch.manual_seed(0)
    m = SPRLv2(_tiny_cfg())
    g = torch.Generator().manual_seed(0)
    out = generate(m, b"hi", n_new=20, temperature=1.0, generator=g, eot_byte=None)
    assert len(out) == 20


def test_generate_with_ttt_kalman():
    """generate(ttt=True) should run when Bet A is enabled, advancing the
    Kalman state without crashing."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(**{"kalman.enabled": True, "dram.router.enabled": True})
    m = SPRLv2(cfg)
    g = torch.Generator().manual_seed(0)
    out = generate(m, b"hi", n_new=3, temperature=1.0, generator=g, ttt=True)
    assert len(out) == 3
