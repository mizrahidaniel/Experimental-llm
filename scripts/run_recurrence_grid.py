"""Recurrence pilot grid — sweep (target_mean_k, max_k) at 100M scale.

Per spec §4.5.3, do not hard-code target_mean_k=3. Pilot the four corners
of {2,3} × {4,6} on 1B tokens each (~4h on a 5080 BF16). Compare validation
loss at matched FLOPs and pick the configuration that fits the 7-day budget.

This script doesn't actually train — it builds the four configs, validates
each, prints the per-config FLOP / memory estimate, and (optionally) drives
N synthetic steps each so a human can sanity-check the loss curves.

Real wall-clock pilots replace `make_synthetic_batch` with a real loader.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from sprl.bench.memory_scaling import (
    estimate_compute,
    estimate_memory_breakdown,
)
from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.training.optim import build_optimizer
from sprl.training.train import train_one_step
from sprl.utils.seeds import seed_everything


GRID = [(2, 4), (2, 6), (3, 4), (3, 6)]


def _synth_batch(cfg, batch=2, n_patches=16):
    return {
        "patch_emb": torch.randn(batch, n_patches, cfg.d_model),
        "byte_targets": torch.randint(
            0, cfg.vocab_size, (batch, n_patches, cfg.patcher.max_patch_bytes)
        ),
        "mtp_targets": torch.randint(0, cfg.vocab_size, (batch, n_patches, cfg.mtp.depth)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_config", default="configs/recommended_100m_pilot.yaml")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--out", default="runs/recurrence_grid")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for target_k, max_k in GRID:
        seed_everything(1234)
        cfg = SPRLConfig.from_yaml(args.base_config)
        cfg.dram.k_iterations_default = target_k
        cfg.dram.k_max = max_k
        cfg.dram.router.k_target = target_k
        cfg.dram.router.k_max = max_k
        cfg.validate()

        model = SPRLv2(cfg)
        compute = estimate_compute(model, mean_recurrent_iterations=float(target_k))
        memory = estimate_memory_breakdown(model, batch=cfg.training.batch_size,
                                           seq_len_patches=cfg.training.seq_len_patches)

        # Drive a few synthetic steps so the optimizer + scheduler are exercised.
        optim = build_optimizer(model, lr=cfg.training.lr)
        losses = []
        for s in range(args.steps):
            m = train_one_step(model, _synth_batch(cfg), optim, cfg, step=s)
            losses.append(m.get("lm", float("nan")))

        row = {
            "target_mean_k": target_k,
            "max_k": max_k,
            "n_params_total_M": compute["n_params_total"] / 1e6,
            "active_params_M": compute["active_params_per_token"] / 1e6,
            "effective_active_M": compute["effective_active_params_per_token"] / 1e6,
            "effective_GFLOPs_per_token": compute["effective_flops_per_token"] / 1e9,
            "estimated_total_GB": memory["estimated_total_gb"],
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        }
        summary.append(row)
        print(json.dumps(row, indent=2))

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
