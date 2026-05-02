"""v3.1 hard validation checks (§8.1)."""

import pytest

from sprl.config import SPRLConfig


def _v3_config_skeleton(variant: str = "recommended") -> SPRLConfig:
    """Minimal config that PASSES v3.1 validation by default."""
    cfg = SPRLConfig(variant=variant)
    cfg.dram.cell_type = "recurrent_middle_block"
    cfg.dram.prefix_layers = 2
    cfg.dram.recurrent_cell_layers = 4
    cfg.dram.suffix_layers = 2
    cfg.dram.routing_granularity = "per_block"
    cfg.kalman.enabled = True
    cfg.kalman.fusion_gate_init = -3.0
    if variant in ("recommended", "recommended_plus_tropical_probe"):
        cfg.tokenizer.type = "bpe"
        cfg.tokenizer.source = "tiny_bpe"
        cfg.vocab_size = 32_000
        cfg.patcher.enabled = False
    return cfg


def test_recommended_skeleton_validates():
    _v3_config_skeleton().validate()


def test_v3_requires_middle_cell():
    cfg = _v3_config_skeleton()
    cfg.dram.cell_type = "uniform_layers"
    with pytest.raises(ValueError, match="recurrent_middle_block"):
        cfg.validate()


def test_v3_requires_per_block_routing():
    cfg = _v3_config_skeleton()
    cfg.dram.routing_granularity = "per_token"
    with pytest.raises(ValueError, match="per_block"):
        cfg.validate()


def test_middle_cell_requires_positive_cell_layers():
    cfg = SPRLConfig(variant="legacy")
    cfg.dram.cell_type = "recurrent_middle_block"
    cfg.dram.recurrent_cell_layers = 0
    with pytest.raises(ValueError, match="recurrent_cell_layers"):
        cfg.validate()


def test_kalman_authority_requires_router_enabled():
    cfg = _v3_config_skeleton()
    cfg.kalman.can_control_router = True
    cfg.dram.router.enabled = False
    with pytest.raises(ValueError, match="can_control_router"):
        cfg.validate()


def test_recommended_disallows_tropical():
    cfg = _v3_config_skeleton(variant="recommended")
    cfg.attention.tropical.enabled = True
    with pytest.raises(ValueError, match="Tropical"):
        cfg.validate()


def test_recommended_disallows_rg_loss():
    cfg = _v3_config_skeleton(variant="recommended")
    cfg.dram.rg_flow.enabled = True
    with pytest.raises(ValueError, match="RG-flow"):
        cfg.validate()


def test_tropical_probe_variant_allows_tropical():
    cfg = _v3_config_skeleton(variant="recommended_plus_tropical_probe")
    cfg.attention.tropical.enabled = True
    cfg.validate()  # no raise


def test_flop_convention_must_be_6N():
    cfg = SPRLConfig(variant="legacy")
    cfg.compute.flop_convention = "18N_per_token_split"
    with pytest.raises(ValueError, match="6N_per_token"):
        cfg.validate()


def test_air_training_mode_must_be_known():
    cfg = SPRLConfig(variant="legacy")
    cfg.dram.router.enabled = True
    cfg.kalman.enabled = True  # router needs kalman to be active path
    cfg.dram.router.training_mode = "wishful_thinking"
    with pytest.raises(ValueError, match="training_mode"):
        cfg.validate()


def test_passive_probe_requires_fusion_gate_init():
    cfg = SPRLConfig(variant="legacy")
    cfg.kalman.enabled = True
    cfg.kalman.mode = "passive_probe"
    cfg.kalman.fusion_gate_init = None
    with pytest.raises(ValueError, match="fusion_gate_init"):
        cfg.validate()


def test_kl_direction_must_be_named():
    cfg = SPRLConfig(variant="legacy")
    cfg.patcher.enabled = False
    cfg.vocab_size = 128_000
    cfg.training.distill_enabled = True
    cfg.training.distill_kl_direction = "magic"
    with pytest.raises(ValueError, match="distill_kl_direction"):
        cfg.validate()


def test_topk_storage_requires_logsumexp():
    cfg = SPRLConfig(variant="legacy")
    cfg.patcher.enabled = False
    cfg.vocab_size = 128_000
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "teacher_token_kl"
    cfg.training.distill_store_teacher_logsumexp = False
    with pytest.raises(ValueError, match="logsumexp"):
        cfg.validate()


def test_topk_storage_requires_indices_and_logits():
    cfg = SPRLConfig(variant="legacy")
    cfg.patcher.enabled = False
    cfg.vocab_size = 128_000
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "teacher_token_kl"
    cfg.training.distill_store_indices = False
    with pytest.raises(ValueError, match="distill_store_indices"):
        cfg.validate()
