"""FineWeb-Edu byte stream.

Tries `datasets.load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)`. If
HuggingFace `datasets` is unavailable or `offline=True`, falls back to a small
Markov chain over common English bigrams so unit tests run anywhere.
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

# Common English bigrams (rough frequency-weighted snapshot, lowercased).
_BIGRAMS = [
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
    "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
    "ra", "ce", "li", "ch", "ll", "be", "ma", "si", "om", "ur",
]
_STARTS = [
    "The ", "A ", "In ", "When ", "If ", "While ", "After ", "For ",
    "Although ", "However, ", "Researchers ", "Students ",
]
_TAILS = [".", "?", "!", ";", ":", ","]


def _markov_english_stream(seed: int) -> Iterator[bytes]:
    """Infinite English-like byte stream, deterministic given seed."""
    rng = random.Random(seed)
    while True:
        out = []
        out.append(rng.choice(_STARTS))
        n_words = rng.randint(6, 20)
        for w in range(n_words):
            if w > 0:
                out.append(" ")
            wlen = rng.randint(1, 4)
            for _ in range(wlen):
                out.append(rng.choice(_BIGRAMS))
        out.append(rng.choice(_TAILS))
        out.append(" ")
        yield "".join(out).encode("utf-8")


def _try_hf_stream(
    dataset_name: str,
    split: str,
    text_key: str,
    seed: int,
    max_examples: Optional[int],
) -> Optional[Iterator[bytes]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return None
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=1024)
    except Exception:
        return None

    def gen() -> Iterator[bytes]:
        for i, item in enumerate(ds):
            if max_examples is not None and i >= max_examples:
                return
            txt = item.get(text_key)
            if txt is None:
                continue
            if isinstance(txt, str):
                yield txt.encode("utf-8", errors="replace")
            elif isinstance(txt, (bytes, bytearray)):
                yield bytes(txt)

    return gen()


def fineweb_edu_stream(
    *,
    seed: int = 0,
    use_hf: bool = True,
    max_examples: Optional[int] = None,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    split: str = "train",
    text_key: str = "text",
) -> Iterator[bytes]:
    """Yield raw bytes one document at a time."""
    if use_hf:
        hf = _try_hf_stream(dataset_name, split, text_key, seed, max_examples)
        if hf is not None:
            yield from hf
            return
    n = 0
    for piece in _markov_english_stream(seed):
        if max_examples is not None and n >= max_examples:
            return
        n += 1
        yield piece
