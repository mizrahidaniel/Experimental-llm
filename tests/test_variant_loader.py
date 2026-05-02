"""Variant resolver maps shortcuts to YAML configs and validates the result."""

import pytest

from sprl.config import SPRLConfig
from sprl.variants import resolve_variant_config, VARIANT_FULL, VARIANT_PILOT


def test_default_variant_is_recommended():
    path = resolve_variant_config(variant=None, config_path=None, full_run=False)
    assert path == VARIANT_PILOT["recommended"]


def test_explicit_config_overrides_variant():
    # The fallback config exists and is unrelated to the recommended pilot.
    path = resolve_variant_config(
        variant="recommended",
        config_path="configs/fallback_boring.yaml",
        full_run=False,
    )
    assert path == "configs/fallback_boring.yaml"


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        resolve_variant_config("notarealthing", None)


def test_full_vs_pilot_route_to_different_files():
    p = resolve_variant_config("recommended", None, full_run=False)
    f = resolve_variant_config("recommended", None, full_run=True)
    assert p != f


@pytest.mark.parametrize("variant,full", [
    ("recommended", False),
    ("recommended", True),
    ("original_bets_mini", False),
    ("original_bets_mini", True),
    ("fallback_boring", False),
    ("fallback_boring", True),
])
def test_all_shipped_variants_load_and_validate(variant, full):
    path = resolve_variant_config(variant, None, full_run=full)
    cfg = SPRLConfig.from_yaml(path)  # raises on validation failure
    assert cfg.d_model > 0
