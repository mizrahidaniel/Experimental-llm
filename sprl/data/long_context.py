"""Long-context byte stream.

Targets exactly `seq_bytes` bytes per sequence (default 64K) so RULER@32K /
@128K can be measured. PG19-style fallback emits long English with structured
chapter markers ("CHAPTER N\\n\\n...") so semantic structure spans 32K tokens.
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

from sprl.data.fineweb_edu import _markov_english_stream, _try_hf_stream


def _padded_long_doc_stream(seed: int, seq_bytes: int) -> Iterator[bytes]:
    rng = random.Random(seed ^ 0x10A6)
    base = _markov_english_stream(seed)
    chapter_n = 0
    buf = bytearray()
    while True:
        while len(buf) < seq_bytes:
            chapter_n += 1
            header = f"\n\nCHAPTER {chapter_n}\n\n".encode("utf-8")
            buf.extend(header)
            for _ in range(rng.randint(16, 32)):
                buf.extend(next(base))
        head = bytes(buf[:seq_bytes])
        del buf[:seq_bytes]
        yield head


def long_context_stream(
    *,
    seed: int = 0,
    seq_bytes: int = 64 * 1024,
    use_hf: bool = True,
    max_examples: Optional[int] = None,
    dataset_name: str = "deepmind/pg19",
    split: str = "train",
    text_key: str = "text",
) -> Iterator[bytes]:
    """Yield exactly `seq_bytes`-long byte sequences."""
    if use_hf:
        hf = _try_hf_stream(dataset_name, split, text_key, seed, max_examples=None)
        if hf is not None:
            buf = bytearray()
            n = 0
            for piece in hf:
                buf.extend(piece)
                while len(buf) >= seq_bytes:
                    head = bytes(buf[:seq_bytes])
                    del buf[:seq_bytes]
                    yield head
                    n += 1
                    if max_examples is not None and n >= max_examples:
                        return
            return
    n = 0
    for piece in _padded_long_doc_stream(seed, seq_bytes):
        if max_examples is not None and n >= max_examples:
            return
        n += 1
        yield piece
