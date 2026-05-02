"""Synthetic Phi-4 + rStar-Math fixture loader tests."""

from __future__ import annotations

import json
from pathlib import Path

from sprl.data.rstar_math import rstar_math_stream
from sprl.data.synthetic_phi4 import (
    _default_fixture_path,
    iter_jsonl,
    synthetic_phi4_stream,
)


def _take(it, n):
    out = []
    for i, x in enumerate(it):
        if i >= n:
            break
        out.append(x)
    return out


def test_default_fixture_exists():
    p = _default_fixture_path()
    assert p.exists(), f"Missing fixture: {p}"
    # At least 4 records.
    n = sum(1 for _ in iter_jsonl([p]))
    assert n >= 4


def test_synthetic_phi4_emits_structured_bytes():
    items = _take(synthetic_phi4_stream(repeat=False, max_examples=4), 4)
    assert len(items) >= 1
    for x in items:
        assert isinstance(x, bytes)
        assert b"<persona>" in x
        assert b"<prompt>" in x
        assert b"<completion>" in x


def test_synthetic_phi4_repeat_extends_indefinitely():
    items = _take(synthetic_phi4_stream(repeat=True, max_examples=50), 50)
    assert len(items) == 50  # bounded only by max_examples


def test_rstar_math_emits_structured_bytes():
    items = _take(rstar_math_stream(repeat=False, max_examples=4), 4)
    assert len(items) >= 1
    for x in items:
        assert isinstance(x, bytes)
        assert b"Problem:" in x
        assert b"Reasoning:" in x
        assert b"Answer:" in x


def test_synthetic_phi4_custom_path(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps({
        "persona": "x",
        "prompt": "y",
        "completion": "z",
        "verifier": "passes",
    }) + "\n")
    items = _take(synthetic_phi4_stream(paths=[p], repeat=False), 5)
    assert len(items) == 1
    assert b"<persona>x</persona>" in items[0]
