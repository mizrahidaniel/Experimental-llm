"""All shipped v3.1 variants load and validate."""

import pytest

from sprl.config import SPRLConfig
from sprl.variants import VARIANT_FULL, VARIANT_PILOT, resolve_variant_config


def test_recommended_plus_tropical_probe_in_pilot_table():
    assert "recommended_plus_tropical_probe" in VARIANT_PILOT


def test_v3_pilot_configs_load_and_validate():
    for variant in ("recommended", "recommended_plus_tropical_probe", "original_bets_mini"):
        path = resolve_variant_config(variant, None, full_run=False)
        cfg = SPRLConfig.from_yaml(path)
        if variant == "recommended":
            assert cfg.variant == "recommended"
            assert cfg.attention.tropical.enabled is False
        if variant == "recommended_plus_tropical_probe":
            assert cfg.variant == "recommended_plus_tropical_probe"
            assert cfg.attention.tropical.enabled is True
            assert cfg.attention.tropical.num_tropical_heads_total == 1
        if variant == "original_bets_mini":
            assert cfg.attention.tropical.enabled is True
            # ≤ 1/8 of heads at mini scale.
            assert cfg.attention.tropical.never_exceed_fraction <= 0.125 + 1e-9
            assert cfg.attention.tropical.num_tropical_heads_total <= 2


def test_v3_full_recommended_loads_and_validates():
    cfg = SPRLConfig.from_yaml(VARIANT_FULL["recommended"])
    assert cfg.variant == "recommended"
    assert cfg.dram.cell_type == "recurrent_middle_block"
    assert cfg.dram.routing_granularity == "per_block"
    assert cfg.attention.tropical.enabled is False
    assert cfg.dram.rg_flow.enabled is False


def test_recommended_300m_targets_realistic_token_count():
    """Sanity check: total_steps × batch × seq_len_patches × ~6 bytes/patch
    should land in the right order of magnitude. Exact total_steps gets tuned
    after μP search and the first throughput measurement; the YAML has a
    starter value within an order of magnitude of the 30B-token target."""
    cfg = SPRLConfig.from_yaml(VARIANT_FULL["recommended"])
    patches = (
        cfg.training.total_steps
        * cfg.training.batch_size
        * cfg.training.seq_len_patches
    )
    # ~5B patches = ~30B bytes (avg 6 bytes/patch). Allow 0.5B - 50B patches.
    assert 5e8 < patches <= 5e10
