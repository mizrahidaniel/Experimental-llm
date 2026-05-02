"""Compute accounting must respect recurrent depth.

`effective_flops_per_token = active_params × mean_recurrent_iterations × 6`
when `count_recurrence_in_active_flops=true`. Validation errors out if
recurrence is on but accounting ignores it.
"""

import pytest

from sprl.config import SPRLConfig


def test_recurrence_with_accounting_off_errors():
    cfg = SPRLConfig()
    cfg.dram.enabled = True
    cfg.dram.k_iterations_default = 3
    cfg.training.count_recurrence_in_active_flops = False
    with pytest.raises(ValueError, match="ignores recurrence"):
        cfg.validate()


def test_recurrence_with_accounting_on_ok():
    cfg = SPRLConfig()
    cfg.dram.enabled = True
    cfg.dram.k_iterations_default = 3
    cfg.training.count_recurrence_in_active_flops = True
    cfg.validate()


def test_no_recurrence_with_accounting_off_ok():
    """k=1 means no recurrence, so accounting ignoring it is fine."""
    cfg = SPRLConfig()
    cfg.dram.k_iterations_default = 1
    cfg.training.count_recurrence_in_active_flops = False
    cfg.validate()


def test_estimate_compute_multiplies_by_iterations():
    """The reported effective FLOPs must be ≈ 6 × params × mean_k."""
    from sprl.bench.memory_scaling import estimate_compute
    from sprl.model import SPRLv2

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
    cfg.dram.k_iterations_default = 3
    cfg.training.count_recurrence_in_active_flops = True
    cfg.validate()

    model = SPRLv2(cfg)
    one_step = estimate_compute(model, mean_recurrent_iterations=1.0)
    three_step = estimate_compute(model, mean_recurrent_iterations=3.0)
    assert three_step["effective_flops_per_token"] == pytest.approx(
        3.0 * one_step["effective_flops_per_token"]
    )
    # active_params is the same regardless of recurrence (standard convention).
    assert one_step["active_params_per_token"] == three_step["active_params_per_token"]
