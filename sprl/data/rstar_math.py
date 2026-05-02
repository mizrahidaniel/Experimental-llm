"""rStar-Math verified MCTS reasoning trace consumer.

JSONL schema (one object per line):

    {
      "problem": "What is 7*8 + 3?",
      "trace": ["7*8 = 56", "56 + 3 = 59"],
      "verified_answer": "59"
    }

Output byte format mirrors a chain-of-thought transcript so the model sees
problem -> stepwise solution -> final answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from sprl.data.synthetic_phi4 import iter_jsonl

PathLike = Union[str, Path]


def _format(rec: dict) -> bytes:
    problem = str(rec.get("problem", "")).strip()
    trace = rec.get("trace", []) or []
    answer = str(rec.get("verified_answer", "")).strip()
    lines = [f"Problem: {problem}", "Reasoning:"]
    for i, step in enumerate(trace, 1):
        lines.append(f"  {i}. {str(step).strip()}")
    lines.append(f"Answer: {answer}")
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8", errors="replace")


def _default_fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "fixtures" / "rstar_math.jsonl"


def rstar_math_stream(
    *,
    paths: Optional[Iterable[PathLike]] = None,
    seed: int = 0,
    repeat: bool = True,
    max_examples: Optional[int] = None,
) -> Iterator[bytes]:
    if paths is None:
        paths = [_default_fixture_path()]
    paths = list(paths)
    n = 0
    while True:
        for rec in iter_jsonl(paths):
            if max_examples is not None and n >= max_examples:
                return
            n += 1
            yield _format(rec)
        if not repeat:
            return
