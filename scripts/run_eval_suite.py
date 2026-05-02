"""Run every eval module against a config + (optional) checkpoint.

Usage:
  python scripts/run_eval_suite.py --config configs/pilot_200m_bf16.yaml \
      --checkpoint path/to/ckpt.pt --evals all --output results.json

The script uses the *fixture* path for HF-gated evals by default; pass
`--use-hf` to switch each module to its HuggingFace dataset (requires the
`datasets` package and network access).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch

from sprl.config import SPRLConfig
from sprl.eval.gpqa import GPQAConfig, evaluate_gpqa
from sprl.eval.gsm8k import GSM8KConfig, evaluate_gsm8k
from sprl.eval.humaneval import HumanEvalConfig, evaluate_humaneval
from sprl.eval.mmlu import MMLUConfig, evaluate_mmlu
from sprl.eval.perplexity import PerplexityConfig, evaluate_perplexity
from sprl.eval.ruler import RULERConfig, evaluate_ruler
from sprl.eval.state_tracking import S5Config, evaluate_s5_state_tracking
from sprl.model import SPRLv2
from sprl.utils.seeds import seed_everything


_ALL_EVALS = ["perplexity", "mmlu", "gsm8k", "humaneval", "gpqa", "ruler", "s5"]


def _load(cfg_path: str, ckpt_path: str | None) -> SPRLv2:
    cfg = (
        SPRLConfig.from_yaml(cfg_path)
        if Path(cfg_path).exists()
        else SPRLConfig()
    )
    model = SPRLv2(cfg)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/pilot_200m_bf16.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--evals", type=str, default="all", help=f"comma-separated list, or 'all'. Choices: {_ALL_EVALS}")
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    p.add_argument("--use-hf", action="store_true", help="Use HuggingFace datasets where available.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ppl-stream", type=str, default="The quick brown fox jumps over the lazy dog. " * 16)
    p.add_argument("--max-questions", type=int, default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    model = _load(args.config, args.checkpoint)
    print(f"Loaded SPRLv2: {model.num_params() / 1e6:.1f}M params", file=sys.stderr)

    if args.evals == "all":
        evals = list(_ALL_EVALS)
    else:
        evals = [e.strip() for e in args.evals.split(",") if e.strip()]

    results: Dict[str, dict] = {}
    for name in evals:
        t0 = time.perf_counter()
        try:
            if name == "perplexity":
                m = evaluate_perplexity(model, args.ppl_stream, PerplexityConfig())
            elif name == "mmlu":
                m = evaluate_mmlu(
                    model,
                    MMLUConfig(use_hf=args.use_hf, max_questions=args.max_questions),
                )
            elif name == "gsm8k":
                m = evaluate_gsm8k(
                    model,
                    GSM8KConfig(use_hf=args.use_hf, max_questions=args.max_questions),
                )
            elif name == "humaneval":
                m = evaluate_humaneval(
                    model,
                    HumanEvalConfig(use_hf=args.use_hf, max_problems=args.max_questions),
                )
            elif name == "gpqa":
                m = evaluate_gpqa(
                    model,
                    GPQAConfig(use_hf=args.use_hf, max_questions=args.max_questions),
                )
            elif name == "ruler":
                m = evaluate_ruler(model, RULERConfig())
            elif name == "s5":
                m = evaluate_s5_state_tracking(model, S5Config())
            else:
                print(f"unknown eval '{name}', skipping", file=sys.stderr)
                continue
        except Exception as exc:
            m = {"error": str(exc)}
        dt = time.perf_counter() - t0
        m["wall_clock_s"] = dt
        results[name] = m
        print(f"[{name}] ({dt:.1f}s) {m}", file=sys.stderr)

    out_blob = {"config": args.config, "checkpoint": args.checkpoint, "results": results}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out_blob, f, indent=2)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        json.dump(out_blob, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
