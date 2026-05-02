"""DCLM (Dolma-Common-LM) byte stream.

HF-or-fallback. Fallback yields a noisier English stream than FineWeb-Edu
(simulating crawl-quality variance) so tests can distinguish the two domains
empirically.
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

from sprl.data.fineweb_edu import _markov_english_stream, _try_hf_stream


_NOISE_TOKENS = [
    " http://example.com/page", " <p>", " </p>", " &nbsp;", " [edit]",
    " (citation needed)", " ::", " >>>", " ... ", "  ",
]


def _noisy_english_stream(seed: int) -> Iterator[bytes]:
    rng = random.Random(seed ^ 0xDC1)
    base = _markov_english_stream(seed)
    while True:
        chunk = next(base)
        if rng.random() < 0.3:
            chunk = chunk + rng.choice(_NOISE_TOKENS).encode("utf-8")
        yield chunk


def dclm_stream(
    *,
    seed: int = 0,
    use_hf: bool = True,
    max_examples: Optional[int] = None,
    dataset_name: str = "mlfoundations/dclm-baseline-1.0",
    split: str = "train",
    text_key: str = "text",
) -> Iterator[bytes]:
    if use_hf:
        hf = _try_hf_stream(dataset_name, split, text_key, seed, max_examples)
        if hf is not None:
            yield from hf
            return
    n = 0
    for piece in _noisy_english_stream(seed):
        if max_examples is not None and n >= max_examples:
            return
        n += 1
        yield piece
