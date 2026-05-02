"""S5 (symmetric group on 5 elements) state-tracking evaluator.

Kill-criterion diagnostic for Bet C (RG-flow depth recurrence). The setup is
the canonical TC^0/NC^1 separation probe: present a sequence of permutations
in S_5 and ask the model to predict the composed permutation. Transformers
are known to fail at long sequences without depth recurrence; SPRL with a
non-trivial T_b should improve here (kill criterion: +10% over baseline).

We render each permutation as 5 digits (the image of (1,2,3,4,5)) and ask the
model to produce the composed result. Scoring is exact match on the 5-digit
output.

We do NOT require any trained capability — the eval reports accuracy on a
configurable seq length so callers can compare configs (Bet C on vs off).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from sprl.eval.byte_io import generate_bytes, score_continuation
from sprl.model import SPRLv2


# ---------------------------------------------------------------------------


def _all_perms() -> List[Tuple[int, ...]]:
    return list(permutations(range(5)))


def _compose(p: Tuple[int, ...], q: Tuple[int, ...]) -> Tuple[int, ...]:
    """(p ∘ q)(i) = p(q(i)). Permutations as tuples mapping i -> p[i]."""
    return tuple(p[q[i]] for i in range(5))


def _format_perm(p: Tuple[int, ...]) -> str:
    return "".join(str(x + 1) for x in p)  # 1-indexed digits


def _parse_perm(s: str) -> Optional[Tuple[int, ...]]:
    digits = [c for c in s if c.isdigit()]
    if len(digits) < 5:
        return None
    take = digits[:5]
    try:
        vals = tuple(int(c) - 1 for c in take)
    except ValueError:
        return None
    if sorted(vals) != list(range(5)):
        return None
    return vals


# ---------------------------------------------------------------------------


@dataclass
class S5Config:
    seq_lengths: List[int] = field(default_factory=lambda: [4, 8, 16])
    n_per_length: int = 8
    seed: int = 0
    method: str = "score"   # "score" = closed-set ranking, "generate" = open
    max_new_bytes: int = 8


def _make_example(rng: random.Random, length: int) -> Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
    perms_pool = _all_perms()
    seq = [tuple(rng.choice(perms_pool)) for _ in range(length)]
    composed = seq[0]
    for p in seq[1:]:
        composed = _compose(composed, p)
    return seq, composed


def _format_prompt(seq: List[Tuple[int, ...]]) -> str:
    rendered = " ".join(_format_perm(p) for p in seq)
    return f"Compose: {rendered}\nResult:"


# ---------------------------------------------------------------------------


def evaluate_s5_state_tracking(
    model: SPRLv2, config: Optional[S5Config] = None
) -> Dict[str, float]:
    """Returns per-length accuracy and overall accuracy.

    Two scoring modes:
      - 'score': rank all 120 permutations by length-normalized log-prob; pick
        the argmax. Robust on tiny untrained models.
      - 'generate': greedy-decode and parse the first 5 digits.
    """
    cfg = config or S5Config()
    model.eval()
    rng = random.Random(cfg.seed)
    all_perms = _all_perms()
    rendered_perms = [_format_perm(p) for p in all_perms]

    out: Dict[str, float] = {}
    overall_correct = 0
    overall_total = 0

    for length in cfg.seq_lengths:
        n_correct = 0
        for _ in range(cfg.n_per_length):
            seq, gold = _make_example(rng, length)
            prompt = _format_prompt(seq) + " "
            if cfg.method == "score":
                # Rank all 120 permutation strings.
                best_idx = 0
                best_score = float("-inf")
                for i, cand in enumerate(rendered_perms):
                    r = score_continuation(model, prompt, cand)
                    s = r.sum_logprob / max(r.n_bytes, 1)
                    if s > best_score:
                        best_score = s
                        best_idx = i
                pred = all_perms[best_idx]
            else:
                gen = generate_bytes(
                    model, prompt, max_new_bytes=cfg.max_new_bytes, stop="\n"
                )
                parsed = _parse_perm(gen)
                pred = parsed if parsed is not None else (0, 1, 2, 3, 4)
            if pred == gold:
                n_correct += 1
        key = f"acc_L{length}"
        out[key] = n_correct / max(cfg.n_per_length, 1)
        overall_correct += n_correct
        overall_total += cfg.n_per_length

    out["accuracy"] = overall_correct / max(overall_total, 1)
    out["n_examples"] = float(overall_total)
    out["chance"] = 1.0 / 120.0
    return out
