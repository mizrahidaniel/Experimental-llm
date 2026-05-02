"""BPE tokenizer wrapper.

Tries HuggingFace `transformers.AutoTokenizer.from_pretrained(source)` first;
falls back to `MiniBPE` if `transformers` is missing or the download fails
(offline, gated repo, etc.). Either way returns an object with `encode`,
`decode`, and `vocab_size` so the rest of the pipeline doesn't care which
backend it got.

Real training needs the real teacher tokenizer. The fallback exists so unit
tests run on CPU with no network.
"""

from __future__ import annotations

from typing import List, Optional

from sprl.tokenizer.mini_bpe import MiniBPE


# Known teacher tokenizers and their vocab sizes (used both as a registry
# of "real" sources and as the canonical vocab_size when offline fallback
# kicks in).
TEACHER_REGISTRY: dict[str, dict] = {
    "llama_3_1_8b": {
        "hf_repo": "meta-llama/Llama-3.1-8B",
        "vocab_size": 128_000,
    },
    "llama_3_1_8b_instruct": {
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "vocab_size": 128_000,
    },
    "qwen_2_5_7b": {
        "hf_repo": "Qwen/Qwen2.5-7B",
        "vocab_size": 152_064,
    },
    "qwen_2_5_7b_instruct": {
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "vocab_size": 152_064,
    },
    "tiny_bpe": {
        "hf_repo": None,
        "vocab_size": 32_000,
    },
}


class _HFTokenizer:
    """Adapter so we expose the same surface as MiniBPE."""

    def __init__(self, hf_tokenizer):
        self._hf = hf_tokenizer
        self.vocab_size = int(getattr(hf_tokenizer, "vocab_size", 0)) or len(hf_tokenizer)
        self.pad_token_id = getattr(hf_tokenizer, "pad_token_id", None) or 0
        self.eos_token_id = getattr(hf_tokenizer, "eos_token_id", None) or 0

    def encode(self, text: str) -> List[int]:
        out = self._hf.encode(text, add_special_tokens=False)
        return list(out)

    def decode(self, ids: List[int]) -> str:
        return self._hf.decode(ids, skip_special_tokens=True)


def load_tokenizer(source: str, allow_network: bool = True):
    """Resolve a teacher source name to a tokenizer object.

    Returns either an `_HFTokenizer` (real HF) or a `MiniBPE` (fallback).
    On `allow_network=False` always returns a MiniBPE — used in unit tests.
    """
    if source not in TEACHER_REGISTRY:
        raise ValueError(
            f"Unknown tokenizer source {source!r}. "
            f"Known: {sorted(TEACHER_REGISTRY)}."
        )
    spec = TEACHER_REGISTRY[source]
    vocab = spec["vocab_size"]

    if not allow_network or spec["hf_repo"] is None:
        return MiniBPE(vocab_size=vocab)

    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        return MiniBPE(vocab_size=vocab)

    try:
        hf = AutoTokenizer.from_pretrained(spec["hf_repo"])
    except Exception:
        # Network failure, gated repo, missing auth, etc. Quietly fall back so
        # CI / offline sandboxes still get a working tokenizer.
        return MiniBPE(vocab_size=vocab)
    return _HFTokenizer(hf)


def vocab_size_for(source: str) -> int:
    """Look up the canonical vocab size for a known teacher source."""
    if source not in TEACHER_REGISTRY:
        raise ValueError(f"Unknown tokenizer source: {source!r}")
    return int(TEACHER_REGISTRY[source]["vocab_size"])
