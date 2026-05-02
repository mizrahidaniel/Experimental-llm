"""Perplexity / bits-per-byte eval."""

import math

from sprl.eval.perplexity import PerplexityConfig, evaluate_perplexity

from tests._eval_helpers import tiny_model


def test_perplexity_returns_finite_metrics():
    model = tiny_model()
    m = evaluate_perplexity(
        model,
        "the quick brown fox jumps over the lazy dog. " * 4,
        PerplexityConfig(chunk_patches=4),
    )
    assert math.isfinite(m["nll"])
    assert math.isfinite(m["ppl"])
    assert math.isfinite(m["bpb"])
    # Untrained byte LM ≈ uniform over 256 → bpb close to 8.
    assert 6.0 < m["bpb"] < 10.0
    assert m["n_bytes"] > 0
    assert m["n_patches"] > 0


def test_perplexity_handles_bytes_input():
    model = tiny_model()
    m = evaluate_perplexity(model, b"abcdefgh" * 4, PerplexityConfig(chunk_patches=2))
    assert m["n_bytes"] == 32


def test_perplexity_handles_empty_stream():
    model = tiny_model()
    m = evaluate_perplexity(model, "", PerplexityConfig())
    assert m["n_bytes"] == 0
    assert m["ppl"] == 1.0


def test_perplexity_max_chunks_truncates():
    model = tiny_model()
    long_stream = "abc" * 200
    m_full = evaluate_perplexity(
        model, long_stream, PerplexityConfig(chunk_patches=2, max_chunks=None)
    )
    m_short = evaluate_perplexity(
        model, long_stream, PerplexityConfig(chunk_patches=2, max_chunks=1)
    )
    assert m_short["n_bytes"] < m_full["n_bytes"]


def test_bpb_is_nll_over_ln2():
    model = tiny_model()
    m = evaluate_perplexity(
        model, "hello world " * 4, PerplexityConfig(chunk_patches=2)
    )
    assert abs(m["bpb"] - m["nll"] / math.log(2)) < 1e-6
