"""Phi-4-style multi-agent synthetic-data consumer.

The *generator* (a multi-agent LLM pipeline that produces persona/prompt/
completion/verifier tuples) is out of scope for this repo. We ship the
consumer — a JSONL loader matching the schema below — plus a tiny on-disk
fixture so unit tests run offline.

Schema (one JSON object per line):

    {
      "persona": "physics tutor explaining gauge symmetry",
      "prompt":  "Explain why electromagnetism has U(1) symmetry.",
      "completion": "Local U(1) phase invariance ...",
      "verifier": "passes"      # optional; we accept any string
    }

The byte stream concatenates the four fields with separators so the patcher
sees a structured but realistic mixture of natural-language formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

PathLike = Union[str, Path]


def _format_record(rec: dict) -> bytes:
    persona = str(rec.get("persona", "")).strip()
    prompt = str(rec.get("prompt", "")).strip()
    completion = str(rec.get("completion", "")).strip()
    verifier = str(rec.get("verifier", "")).strip()
    parts = [
        f"<persona>{persona}</persona>",
        f"<prompt>{prompt}</prompt>",
        f"<completion>{completion}</completion>",
    ]
    if verifier:
        parts.append(f"<verifier>{verifier}</verifier>")
    parts.append("")
    return ("\n".join(parts) + "\n").encode("utf-8", errors="replace")


def iter_jsonl(paths: Iterable[PathLike]) -> Iterator[dict]:
    for p in paths:
        p = Path(p)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _default_fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "fixtures" / "synthetic_phi4.jsonl"


def synthetic_phi4_stream(
    *,
    paths: Optional[Iterable[PathLike]] = None,
    seed: int = 0,  # unused; kept for API parity (records are read in order)
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
            yield _format_record(rec)
        if not repeat:
            return
