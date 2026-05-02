"""Long-context loader: exact-length sequences, chapter markers."""

from __future__ import annotations

from sprl.data.long_context import long_context_stream


def _take(it, n):
    out = []
    for i, x in enumerate(it):
        if i >= n:
            break
        out.append(x)
    return out


def test_long_context_emits_exact_length():
    L = 4096
    items = _take(long_context_stream(seed=0, seq_bytes=L, use_hf=False, max_examples=3), 3)
    assert len(items) == 3
    for x in items:
        assert isinstance(x, bytes)
        assert len(x) == L


def test_long_context_chapter_markers_present():
    L = 8192
    items = _take(long_context_stream(seed=0, seq_bytes=L, use_hf=False, max_examples=2), 2)
    blob = b"".join(items)
    assert b"CHAPTER" in blob


def test_long_context_determinism():
    L = 2048
    a = _take(long_context_stream(seed=7, seq_bytes=L, use_hf=False, max_examples=4), 4)
    b = _take(long_context_stream(seed=7, seq_bytes=L, use_hf=False, max_examples=4), 4)
    assert a == b


def test_long_context_64k_shape_is_supported():
    L = 64 * 1024
    items = _take(long_context_stream(seed=0, seq_bytes=L, use_hf=False, max_examples=1), 1)
    assert len(items) == 1
    assert len(items[0]) == L
