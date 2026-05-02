"""Per-block K* routing: K is one int per batch element, not per token."""

import torch

from sprl.config import SPRLConfig
from sprl.model import SPRLv2


def _cfg_middle_cell():
    cfg = SPRLConfig(d_model=64, n_layers=8, max_seq_len_patches=64)
    cfg.attention.mla.n_heads = 4
    cfg.attention.mla.d_c_latent = 32
    cfg.attention.mla.d_qhead = 16
    cfg.attention.mla.d_rope_decoupled = 8
    cfg.moe.num_routed_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 32
    cfg.patcher.byte_encoder_dim = 32
    cfg.patcher.max_patch_bytes = 8
    cfg.dram.cell_type = "recurrent_middle_block"
    cfg.dram.prefix_layers = 1
    cfg.dram.recurrent_cell_layers = 2
    cfg.dram.suffix_layers = 1
    cfg.dram.routing_granularity = "per_block"
    cfg.dram.use_depth_attention = False
    cfg.dram.k_iterations_default = 2
    cfg.dram.k_max = 4
    return cfg


def test_per_block_routing_aggregates_to_batch_dim():
    """Diagnostics report a single mean K across the batch (one int per element,
    averaged for logging) — not a per-token distribution."""
    model = SPRLv2(_cfg_middle_cell())
    z = torch.randn(2, 16, 64)
    out = model.forward_from_patches(z)
    # Both mean and max should be finite scalars in [1, k_max].
    mean_k = out.diagnostics["mean_k"]
    max_k = out.diagnostics["max_k"]
    assert 1.0 <= mean_k <= 4.0
    assert 1.0 <= max_k <= 4.0


def test_per_block_routing_passes_batch_shape_to_middle():
    """Ensure the recurrent cell receives a [B] iteration tensor, not [B, S]."""
    model = SPRLv2(_cfg_middle_cell())
    # Patch middle.forward to capture the k_per_block argument shape.
    captured = {}
    real = model.middle.forward

    def spy(z, k_per_block, return_iterations=False):
        captured["shape"] = tuple(k_per_block.shape)
        return real(z, k_per_block, return_iterations=return_iterations)

    model.middle.forward = spy
    out = model.forward_from_patches(torch.randn(3, 16, 64))
    assert captured["shape"] == (3,)  # one K per batch element


def test_legacy_per_token_routing_unaffected():
    cfg = SPRLConfig(d_model=64, n_layers=2, max_seq_len_patches=64)
    cfg.attention.mla.n_heads = 4
    cfg.attention.mla.d_c_latent = 32
    cfg.attention.mla.d_qhead = 16
    cfg.attention.mla.d_rope_decoupled = 8
    cfg.moe.num_routed_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 32
    cfg.patcher.byte_encoder_dim = 32
    cfg.patcher.max_patch_bytes = 8
    # cell_type=uniform_layers + routing_granularity=per_token (defaults)
    model = SPRLv2(cfg)
    out = model.forward_from_patches(torch.randn(2, 16, 64))
    assert torch.isfinite(out.byte_logits).all()
