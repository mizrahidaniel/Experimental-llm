"""Tokenizer subpackage: registry, MiniBPE fallback, vocab lookup."""

import pytest

from sprl.tokenizer import (
    MiniBPE,
    TEACHER_REGISTRY,
    load_tokenizer,
    vocab_size_for,
)


def test_vocab_size_for_known_sources():
    assert vocab_size_for("llama_3_1_8b") == 128_000
    assert vocab_size_for("qwen_2_5_7b") == 152_064
    assert vocab_size_for("tiny_bpe") == 32_000


def test_vocab_size_unknown_raises():
    with pytest.raises(ValueError, match="Unknown tokenizer source"):
        vocab_size_for("imaginary_model")


def test_mini_bpe_round_trip():
    tok = MiniBPE(vocab_size=32_000)
    text = "hello world"
    ids = tok.encode(text)
    assert all(0 <= i < 256 for i in ids)
    assert tok.decode(ids) == text


def test_mini_bpe_rejects_too_small_vocab():
    with pytest.raises(ValueError, match="vocab_size must be"):
        MiniBPE(vocab_size=128)


def test_load_tokenizer_offline_falls_back_to_minibpe():
    """allow_network=False forces the MiniBPE fallback regardless of source."""
    tok = load_tokenizer("llama_3_1_8b", allow_network=False)
    assert isinstance(tok, MiniBPE)
    assert tok.vocab_size == 128_000


def test_load_tokenizer_tiny_bpe_no_network():
    tok = load_tokenizer("tiny_bpe", allow_network=False)
    assert isinstance(tok, MiniBPE)
    assert tok.vocab_size == 32_000


def test_registry_entries_have_expected_keys():
    for name, spec in TEACHER_REGISTRY.items():
        assert "hf_repo" in spec
        assert "vocab_size" in spec
        assert isinstance(spec["vocab_size"], int) and spec["vocab_size"] > 0
