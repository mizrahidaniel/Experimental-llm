"""Compute estimator must report effective FLOPs = 6 × params × overhead × mean_k."""

import pytest

from sprl.bench.memory_scaling import estimate_compute, estimate_memory_breakdown
from sprl.config import SPRLConfig
from sprl.model import SPRLv2


def _tiny_cfg(cell_type="uniform_layers"):
    cfg = SPRLConfig(d_model=64, n_layers=4, max_seq_len_patches=64)
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
    cfg.dram.cell_type = cell_type
    if cell_type == "recurrent_middle_block":
        cfg.dram.prefix_layers = 1
        cfg.dram.recurrent_cell_layers = 2
        cfg.dram.suffix_layers = 1
        cfg.dram.routing_granularity = "per_block"
    return cfg


def test_overhead_factor_multiplies_flops():
    cfg = _tiny_cfg()
    model = SPRLv2(cfg)

    cfg.compute.architecture_overhead_factor = 1.0
    a = estimate_compute(model, mean_recurrent_iterations=1.0)["effective_flops_per_token"]

    cfg.compute.architecture_overhead_factor = 2.0
    b = estimate_compute(model, mean_recurrent_iterations=1.0)["effective_flops_per_token"]

    assert b == pytest.approx(2.0 * a)


def test_uniform_layout_flops_scale_with_mean_k():
    """For uniform layouts, all params are in the recurrent stack ⇒
    effective_flops scales linearly with mean_k."""
    cfg = _tiny_cfg("uniform_layers")
    cfg.compute.architecture_overhead_factor = 1.0
    cfg.training.count_recurrence_in_active_flops = True
    model = SPRLv2(cfg)
    one = estimate_compute(model, mean_recurrent_iterations=1.0)["effective_flops_per_token"]
    three = estimate_compute(model, mean_recurrent_iterations=3.0)["effective_flops_per_token"]
    assert three == pytest.approx(3.0 * one)


def test_middle_cell_layout_only_cell_scales():
    """For middle-cell, only cell_active scales with K; prefix/suffix don't."""
    cfg = _tiny_cfg("recurrent_middle_block")
    cfg.compute.architecture_overhead_factor = 1.0
    model = SPRLv2(cfg)

    e1 = estimate_compute(model, mean_recurrent_iterations=1.0)
    e3 = estimate_compute(model, mean_recurrent_iterations=3.0)

    # prefix + cell·1 + suffix vs prefix + cell·3 + suffix.
    delta = e3["effective_active_params_per_token"] - e1["effective_active_params_per_token"]
    expected = 2.0 * e1["cell_active_params"]  # added 2 more cell iterations
    assert abs(delta - expected) < 1e-3


def test_memory_breakdown_returns_all_required_keys():
    cfg = _tiny_cfg()
    model = SPRLv2(cfg)
    m = estimate_memory_breakdown(model, batch=2, seq_len_patches=16)
    required = {
        "weights_gb", "gradients_gb", "optimizer_states_gb", "activations_gb",
        "kv_or_attention_cache_gb", "kalman_state_gb", "moe_buffers_gb",
        "teacher_logits_gb", "workspace_gb", "estimated_total_gb",
    }
    assert required.issubset(m.keys())
    assert m["estimated_total_gb"] > 0
