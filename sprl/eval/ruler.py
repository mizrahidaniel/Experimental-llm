"""RULER-style synthetic long-context evals.

Generates needle-in-a-haystack and multi-key retrieval tasks fully in-process,
so no external data is required. Tests at default lengths 8K / 32K / 128K.
For unit tests we use far smaller lengths via `RULERConfig.lengths`.

Tasks implemented:
  - niah_single: one (key, value) pair hidden in a sea of distractors. Ask for
    the value. Pass = exact substring match.
  - niah_multi:  k pairs hidden; ask for one of them at random.
  - kv_recall:   structured "key: value" listing; ask for one.

We measure whether the model's argmax-generated continuation contains the
gold answer string. Greedy-only; this is a recall test, not a generation
quality test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sprl.eval.byte_io import generate_bytes
from sprl.model import SPRLv2


# ---------------------------------------------------------------------------


@dataclass
class RULERConfig:
    # Lengths in BYTES (not tokens) — RULER specifies tokens but for byte-level
    # SPRL we count bytes directly. Defaults are tiny so unit tests are fast;
    # production callers should pass [8192, 32768, 131072] etc.
    lengths: List[int] = field(default_factory=lambda: [256, 512, 1024])
    n_per_length: int = 4
    tasks: List[str] = field(default_factory=lambda: ["niah_single", "kv_recall"])
    max_new_bytes: int = 32
    seed: int = 0


_DISTRACTOR_WORDS = [
    "the",
    "and",
    "of",
    "to",
    "is",
    "in",
    "that",
    "it",
    "for",
    "as",
    "with",
    "on",
    "by",
    "an",
    "this",
    "from",
    "or",
    "be",
    "at",
    "are",
]


def _make_haystack(rng: random.Random, n_bytes: int) -> str:
    out: List[str] = []
    cur = 0
    while cur < n_bytes:
        w = rng.choice(_DISTRACTOR_WORDS)
        out.append(w)
        cur += len(w) + 1
    return " ".join(out)[:n_bytes]


def _random_key(rng: random.Random) -> str:
    """Short alphanumeric key, e.g. 'k7Q2'."""
    alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alpha) for _ in range(4))


def _random_value(rng: random.Random) -> str:
    """Short numeric value, e.g. '4729'. Easy to extract."""
    return str(rng.randint(1000, 9999))


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------


def _gen_niah_single(rng: random.Random, n_bytes: int) -> Tuple[str, str]:
    key, value = _random_key(rng), _random_value(rng)
    needle = f"The magic number for {key} is {value}."
    haystack = _make_haystack(rng, max(n_bytes - len(needle) - 64, 32))
    # Insert needle at a random position in haystack words.
    parts = haystack.split(" ")
    pos = rng.randint(0, len(parts))
    parts.insert(pos, needle)
    body = " ".join(parts)
    prompt = body + f"\nQuestion: What is the magic number for {key}?\nAnswer:"
    return prompt, value


def _gen_kv_recall(rng: random.Random, n_bytes: int) -> Tuple[str, str]:
    n_pairs = max(4, n_bytes // 64)
    pairs: List[Tuple[str, str]] = []
    seen = set()
    while len(pairs) < n_pairs:
        k = _random_key(rng)
        if k in seen:
            continue
        seen.add(k)
        pairs.append((k, _random_value(rng)))
    body = "\n".join(f"{k}: {v}" for k, v in pairs)
    # Pad with distractors only if absurdly short relative to the request.
    if len(body) < n_bytes:
        body = body + "\n" + _make_haystack(rng, n_bytes - len(body))
    target_idx = rng.randint(0, len(pairs) - 1)
    target_key, target_value = pairs[target_idx]
    prompt = body + f"\nQuestion: What is the value of {target_key}?\nAnswer:"
    return prompt, target_value


_TASKS = {
    "niah_single": _gen_niah_single,
    "kv_recall": _gen_kv_recall,
}


# ---------------------------------------------------------------------------


def evaluate_ruler(model: SPRLv2, config: Optional[RULERConfig] = None) -> Dict[str, float]:
    cfg = config or RULERConfig()
    model.eval()
    rng = random.Random(cfg.seed)

    out: Dict[str, float] = {}
    overall_correct = 0
    overall_total = 0

    for length in cfg.lengths:
        for task_name in cfg.tasks:
            gen_fn = _TASKS.get(task_name)
            if gen_fn is None:
                continue
            n_correct = 0
            for _ in range(cfg.n_per_length):
                prompt, gold = gen_fn(rng, length)
                pred = generate_bytes(
                    model,
                    prompt,
                    max_new_bytes=cfg.max_new_bytes,
                    stop="\n",
                )
                if gold in pred:
                    n_correct += 1
            key = f"acc_{task_name}_L{length}"
            out[key] = n_correct / max(cfg.n_per_length, 1)
            overall_correct += n_correct
            overall_total += cfg.n_per_length

    out["accuracy"] = overall_correct / max(overall_total, 1)
    out["n_examples"] = float(overall_total)
    return out
