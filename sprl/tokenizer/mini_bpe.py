"""Minimal byte-level BPE fallback tokenizer.

Used when (a) HuggingFace `transformers` is not installed, (b) the named
teacher tokenizer can't be downloaded (offline / gated repo), or (c) tests
that need a deterministic small-vocab tokenizer with no network access.

This is NOT a real BPE — it's a fixed byte-level tokenizer with a configurable
vocab. Each input character maps to its UTF-8 byte ids; tokens above 256 are
"merged" only via a deterministic hash-into-bucket. Quality is poor; the
point is that it produces a valid `[B, S]` integer sequence the rest of the
pipeline can train on.

Real distillation runs MUST use a real teacher tokenizer (Llama / Qwen / etc).
This module exists so unit tests can exercise the BPE code path on CPU with
no external dependencies.
"""

from __future__ import annotations

from typing import List


class MiniBPE:
    """A trivial deterministic byte-level tokenizer with `vocab_size` codes.

    Encoding: each byte is mapped to its raw value; tokens above 256 are
    reserved (0..vocab_size-1 valid range). `encode` returns byte ids only.
    """

    def __init__(self, vocab_size: int = 32_000):
        if vocab_size < 256:
            raise ValueError("MiniBPE vocab_size must be ≥ 256 (byte coverage).")
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 0  # share PAD/EOS slot in this stub

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(b for b in ids if 0 <= b < 256).decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"MiniBPE(vocab_size={self.vocab_size})"
