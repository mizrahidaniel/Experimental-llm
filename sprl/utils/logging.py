"""Lightweight metric logger.

Wraps `print` by default; can be swapped for wandb/tensorboard later without
touching the training loop. Keeps a per-step dict of metrics so the kill-criterion
diagnostics can inspect history.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


class MetricLogger:
    def __init__(self, log_every: int = 100, jsonl_path: Optional[str] = None):
        self.log_every = log_every
        self.jsonl_path = jsonl_path
        self.history: Dict[str, List[float]] = defaultdict(list)
        self._step = 0
        self._start = time.time()

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        s = self._step if step is None else step
        self._step = s + 1
        for k, v in metrics.items():
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    continue
                self.history[k].append(fv)
            except (TypeError, ValueError):
                # Non-numeric — record as raw, not in history.
                pass

        if self.jsonl_path is not None:
            payload = {"step": s, "wall": time.time() - self._start, **{
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in metrics.items()
            }}
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(payload) + "\n")

        if s % self.log_every == 0:
            parts = [f"step={s}"] + [
                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in metrics.items()
            ]
            print(" ".join(parts), flush=True)

    def latest(self, key: str) -> Optional[float]:
        return self.history[key][-1] if self.history[key] else None
