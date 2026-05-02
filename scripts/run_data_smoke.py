"""Data-pipeline smoke test.

Loads the pilot 200M config, builds a data iterator on the synthetic fallback
streams (no HF download), draws 3 batches, and prints the shapes plus a
histogram of patch sizes so a human can eyeball the patcher's behaviour.

Usage:
    python scripts/run_data_smoke.py
    python scripts/run_data_smoke.py --config configs/pilot_200m_bf16.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running directly from a worktree without installing.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from sprl.config import SPRLConfig, default_pilot_200m  # noqa: E402
from sprl.data import DataConfig, build_data_iterator  # noqa: E402


def _hist(counts: Counter, width: int = 40) -> str:
    if not counts:
        return "(empty)"
    lo, hi = min(counts), max(counts)
    total = sum(counts.values())
    lines = []
    peak = max(counts.values())
    for k in range(lo, hi + 1):
        v = counts.get(k, 0)
        bar = "#" * int(width * v / max(peak, 1))
        pct = 100.0 * v / total if total else 0
        lines.append(f"  size={k:2d} | {bar:<{width}} | n={v} ({pct:.1f}%)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--chunk-bytes", type=int, default=512)
    ap.add_argument("--seq-patches", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.config and Path(args.config).exists():
        cfg = SPRLConfig.from_yaml(args.config)
    else:
        cfg = default_pilot_200m()

    print(f"Loaded config: d_model={cfg.d_model} max_patch_bytes={cfg.patcher.max_patch_bytes} "
          f"threshold={cfg.patcher.entropy_threshold}")

    dc = DataConfig(
        chunk_bytes=args.chunk_bytes,
        max_seq_patches=args.seq_patches,
        batch_size=args.batch_size,
        use_hf=False,
        seed=args.seed,
        max_examples_per_source=200,
    )

    it = build_data_iterator(cfg, data_cfg=dc)

    sizes: Counter[int] = Counter()
    for i in range(args.batches):
        batch = next(it)
        print(f"\n=== batch {i} ===")
        for k, v in batch.items():
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        am = batch["attention_mask"].bool()
        ln = batch["lengths"][am]
        for n in ln.tolist():
            sizes[n] += 1
        n_real = int(am.sum().item())
        avg = float(ln.float().mean().item()) if n_real else 0.0
        print(f"  real patches: {n_real} | avg patch len: {avg:.2f}")

    overall_total = sum(sizes.values())
    overall_avg = sum(k * v for k, v in sizes.items()) / max(overall_total, 1)
    print(f"\nOverall avg patch size across {args.batches} batches: {overall_avg:.2f} "
          f"(spec target ≈ 6.0; depends on byte_lm training).")
    print("\nPatch-size histogram:")
    print(_hist(sizes))


if __name__ == "__main__":
    main()
