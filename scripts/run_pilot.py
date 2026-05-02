"""Run a tiny pilot: instantiate a 200M config, drive synthetic data through it,
print parameter count and one step of loss.

Useful as a smoke test on a CPU-only box. Replace the synthetic loader with a
real BLT-tokenized iterator for an actual training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.training.train import train_one_step
from sprl.training.optim import build_optimizer
from sprl.utils.seeds import seed_everything


def make_synthetic_batch(cfg: SPRLConfig, batch_size: int = 2, n_patches: int = 16) -> dict:
    return {
        "patch_emb": torch.randn(batch_size, n_patches, cfg.d_model),
        "byte_targets": torch.randint(
            0, cfg.vocab_size, (batch_size, n_patches, cfg.patcher.max_patch_bytes)
        ),
        "mtp_targets": torch.randint(0, cfg.vocab_size, (batch_size, n_patches, cfg.mtp.depth)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/pilot_200m_bf16.yaml")
    p.add_argument("--steps", type=int, default=2)
    args = p.parse_args()

    seed_everything(1234)
    cfg = SPRLConfig.from_yaml(args.config) if Path(args.config).exists() else SPRLConfig()
    model = SPRLv2(cfg)
    print(f"Built SPRLv2: {model.num_params() / 1e6:.1f}M params")

    optim = build_optimizer(
        model, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
        optimizer_type=cfg.precision.optimizer_type,
    )

    for step in range(args.steps):
        batch = make_synthetic_batch(cfg, batch_size=2, n_patches=16)
        m = train_one_step(model, batch, optim, cfg, step=step)
        print(f"step {step}: lm={m['lm']:.3f} total={m['total']:.3f} mean_k={m.get('mean_k', 0):.2f}")


if __name__ == "__main__":
    main()
