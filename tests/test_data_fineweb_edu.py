"""FineWeb-Edu / DCLM / StarCoder / MathPile fallback determinism tests.

We exercise the *fallback* (synthetic) branch so tests run offline.
"""

from __future__ import annotations

from sprl.data import (
    dclm_stream,
    fineweb_edu_stream,
    math_pile_stream,
    starcoder_v2_stream,
)


def _take(it, n):
    out = []
    for i, x in enumerate(it):
        if i >= n:
            break
        out.append(x)
    return out


def test_fineweb_fallback_yields_bytes():
    items = _take(fineweb_edu_stream(seed=0, use_hf=False, max_examples=10), 10)
    assert len(items) == 10
    for x in items:
        assert isinstance(x, bytes)
        assert len(x) > 0


def test_fineweb_determinism():
    a = _take(fineweb_edu_stream(seed=42, use_hf=False, max_examples=20), 20)
    b = _take(fineweb_edu_stream(seed=42, use_hf=False, max_examples=20), 20)
    assert a == b


def test_fineweb_seed_changes_output():
    a = _take(fineweb_edu_stream(seed=0, use_hf=False, max_examples=10), 10)
    b = _take(fineweb_edu_stream(seed=1, use_hf=False, max_examples=10), 10)
    assert a != b


def test_dclm_fallback_differs_from_fineweb():
    a = _take(fineweb_edu_stream(seed=0, use_hf=False, max_examples=20), 20)
    b = _take(dclm_stream(seed=0, use_hf=False, max_examples=20), 20)
    # DCLM injects noise tokens with prob 0.3 — at least one example should differ.
    assert any(x != y for x, y in zip(a, b))


def test_starcoder_fallback_looks_codey():
    items = _take(starcoder_v2_stream(seed=0, use_hf=False, max_examples=5), 5)
    blob = b"".join(items)
    assert b"def " in blob
    assert b"return" in blob


def test_math_pile_fallback_has_digits():
    items = _take(math_pile_stream(seed=0, use_hf=False, max_examples=5), 5)
    blob = b"".join(items)
    # Math fallback produces lots of digits and operators.
    assert any(b in blob for b in [b"=", b"+", b"-"])
    assert any(c in blob for c in b"0123456789")


def test_max_examples_bounds_output():
    items = _take(fineweb_edu_stream(seed=0, use_hf=False, max_examples=3), 100)
    assert len(items) == 3
