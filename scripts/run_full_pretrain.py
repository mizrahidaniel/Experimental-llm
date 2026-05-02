"""Full Stage 3 pretraining driver.

Composes the data pipeline (sprl/data/), distillation pipeline (sprl/training/),
training loop (sprl/training/train.py), and eval harness (sprl/eval/) into a
single end-to-end script per spec §6.4.

This is a *thin* orchestrator — the heavy lifting lives in subpackages. The
goal is to be readable: a person should be able to follow the four phases of
Stage 3 (distill / synthetic+RL / long-context / post-training) by reading
this file alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.training.optim import build_optimizer
from sprl.training.train import train_one_step
from sprl.utils.logging import MetricLogger
from sprl.utils.seeds import seed_everything


def build_data(cfg: SPRLConfig, phase: str):
    """Defer to sprl.data.build_data_iterator. If sprl.data is absent (agents
    not yet merged), fall back to a synthetic batcher."""
    try:
        from sprl.data.loader import build_data_iterator  # type: ignore

        return build_data_iterator(cfg, phase=phase)
    except ImportError:
        return _synthetic_iter(cfg)


def _synthetic_iter(cfg):
    while True:
        yield {
            "patch_emb": torch.randn(2, 16, cfg.d_model),
            "byte_targets": torch.randint(0, cfg.vocab_size, (2, 16, cfg.patcher.max_patch_bytes)),
            "mtp_targets": torch.randint(0, cfg.vocab_size, (2, 16, cfg.mtp.depth)),
        }


def build_evals(cfg: SPRLConfig):
    try:
        from sprl.eval.harness import EvalSuite  # type: ignore
        return EvalSuite(cfg)
    except ImportError:
        return None


def phase_distill(model, cfg, optim, log, n_steps: int):
    """Phase 3a: distillation pretraining (~250B tokens in spec)."""
    data = build_data(cfg, phase="distill")
    for step, batch in enumerate(data):
        if step >= n_steps:
            break
        m = train_one_step(model, batch, optim, cfg, step=step)
        log.log({**m, "phase": "distill"}, step=step)


def phase_synthetic(model, cfg, optim, log, n_steps: int, base_step: int = 0):
    """Phase 3b: synthetic + verifier-RL (~250B tokens in spec)."""
    data = build_data(cfg, phase="synthetic")
    for step, batch in enumerate(data):
        if step >= n_steps:
            break
        m = train_one_step(model, batch, optim, cfg, step=base_step + step)
        log.log({**m, "phase": "synthetic"}, step=base_step + step)


def phase_long_context(model, cfg, optim, log, n_steps: int, base_step: int = 0):
    """Phase 3c: long context (~200B tokens, 64K patches)."""
    cfg.max_seq_len_patches = max(cfg.max_seq_len_patches, 64 * 1024)
    data = build_data(cfg, phase="long_context")
    for step, batch in enumerate(data):
        if step >= n_steps:
            break
        m = train_one_step(model, batch, optim, cfg, step=base_step + step)
        log.log({**m, "phase": "long_ctx"}, step=base_step + step)


def phase_post_training(model, cfg, optim, log, n_steps: int, base_step: int = 0):
    """Phase 3d: SFT + DPO + GRPO (~100B tokens equivalent)."""
    data = build_data(cfg, phase="post_training")
    for step, batch in enumerate(data):
        if step >= n_steps:
            break
        m = train_one_step(model, batch, optim, cfg, step=base_step + step)
        log.log({**m, "phase": "post_training"}, step=base_step + step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps_per_phase", type=int, default=10)
    p.add_argument("--out", default="runs/full_pretrain")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = SPRLConfig.from_yaml(args.config)
    seed_everything(cfg.training.seed)

    model = SPRLv2(cfg)
    print(f"Built SPRLv2: {model.num_params() / 1e6:.1f}M params")

    optim = build_optimizer(
        model,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        optimizer_type=cfg.precision.optimizer_type,
    )
    log = MetricLogger(log_every=10, jsonl_path=str(out / "metrics.jsonl"))

    # Run all 4 phases.
    s = args.steps_per_phase
    phase_distill(model, cfg, optim, log, n_steps=s)
    phase_synthetic(model, cfg, optim, log, n_steps=s, base_step=s)
    phase_long_context(model, cfg, optim, log, n_steps=s, base_step=2 * s)
    phase_post_training(model, cfg, optim, log, n_steps=s, base_step=3 * s)

    # Final evaluation.
    evals = build_evals(cfg)
    if evals is not None:
        results = evals.run_all(model)
        print(results)


if __name__ == "__main__":
    main()
