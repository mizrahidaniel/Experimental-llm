"""Variant resolver: maps a `--variant` shortcut to a default config path.

This is intentionally thin — the heavy lifting is done by `SPRLConfig` and
the named YAML files in `configs/`. The variant is just a friendly name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


VARIANT_PILOT = {
    "recommended": "configs/recommended_100m_pilot.yaml",
    "recommended_plus_tropical_probe": "configs/recommended_plus_tropical_probe_pilot.yaml",
    "original_bets_mini": "configs/original_bets_mini_100m_pilot.yaml",
    "fallback_boring": "configs/fallback_boring.yaml",
}

VARIANT_FULL = {
    "recommended": "configs/recommended_300m_full.yaml",
    # No full-run config for the tropical probe variant — by design, run only
    # the pilot to decide whether to fold tropical into the next iteration.
    "original_bets_mini": "configs/original_bets_mini_300m_full.yaml",
    "fallback_boring": "configs/fallback_boring.yaml",
}


def resolve_variant_config(
    variant: Optional[str],
    config_path: Optional[str],
    full_run: bool = False,
) -> str:
    """Pick a config path. Explicit `config_path` wins; otherwise look up by variant."""
    if config_path:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        return config_path
    if not variant:
        # Default to recommended.
        variant = "recommended"
    table = VARIANT_FULL if full_run else VARIANT_PILOT
    if variant not in table:
        raise ValueError(
            f"Unknown variant '{variant}'. Valid: {sorted(table)}. "
            "Override with --config <path> if you have a custom config."
        )
    path = table[variant]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Variant '{variant}' resolves to {path} which does not exist."
        )
    return path


def print_variant_banner(cfg, variant: Optional[str] = None) -> None:
    """One-line per active mechanism — printed at startup so misconfigured runs
    are obvious from the first line of stdout."""
    parts = [
        f"variant={variant or 'recommended'}",
        "tokenizer=BLT" if cfg.patcher.enabled else "tokenizer=raw",
        f"d_model={cfg.d_model}",
        f"n_layers={cfg.n_layers}",
        f"kalman={'on' if cfg.kalman.enabled else 'off'}",
        f"router={'air' if cfg.dram.router.enabled else 'entropy'}",
        f"k={cfg.dram.k_iterations_default}/{cfg.dram.k_max}",
        f"tropical={'on' if cfg.attention.tropical.enabled else 'off'}",
        f"rg_loss={'on' if cfg.dram.rg_flow.enabled else 'off (diag only)'}",
        f"moe={cfg.moe.num_routed_experts}r+{cfg.moe.num_shared_experts}s top-{cfg.moe.active_per_token}",
        f"distill={'on' if cfg.training.distill_enabled else 'off'}",
        f"active_sel={'on' if cfg.training.active_selection_enabled else 'off'}",
        f"precision={cfg.precision.forward_default}",
    ]
    print("[sprl] " + " | ".join(parts), flush=True)
