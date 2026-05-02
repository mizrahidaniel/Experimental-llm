"""Shared helpers for eval/bench tests: build a tiny SPRLv2 quickly."""

from __future__ import annotations

from sprl.config import SPRLConfig
from sprl.model import SPRLv2


def tiny_cfg(**overrides) -> SPRLConfig:
    cfg = SPRLConfig(d_model=64, n_layers=2, max_seq_len_patches=128)
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
    cfg.patcher.max_patch_bytes = 8
    cfg.mtp.depth = 2
    for k, v in overrides.items():
        head = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            head = getattr(head, p)
        setattr(head, parts[-1], v)
    return cfg


def tiny_model() -> SPRLv2:
    m = SPRLv2(tiny_cfg())
    m.eval()
    return m
