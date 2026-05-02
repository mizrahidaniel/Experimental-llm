"""Curriculum: weight updates respect freeze, normalization."""

import random

import pytest

from sprl.training.curriculum import PerplexityCorrelationCurriculum


def test_initial_weights_uniform():
    c = PerplexityCorrelationCurriculum(["a", "b", "c", "d"], freeze_until_tokens=0)
    assert all(abs(w - 0.25) < 1e-9 for w in c.weights.values())


def test_no_update_during_freeze():
    c = PerplexityCorrelationCurriculum(
        ["a", "b"], freeze_until_tokens=10**11, update_every_tokens=10
    )
    before = dict(c.weights)
    c.maybe_update(tokens_seen=100, domain_losses={"a": 1.0, "b": 5.0}, downstream_loss=2.0)
    assert c.weights == before


def test_update_after_freeze_normalizes():
    c = PerplexityCorrelationCurriculum(
        ["a", "b", "c"], freeze_until_tokens=0, update_every_tokens=0, smoothing=0.0
    )
    c.maybe_update(
        tokens_seen=10**11,
        domain_losses={"a": 2.0, "b": 4.0, "c": 8.0},
        downstream_loss=2.0,
    )
    s = sum(c.weights.values())
    assert abs(s - 1.0) < 1e-6
    # Domain "a" tracks downstream loss best ⇒ highest weight.
    assert c.weights["a"] == max(c.weights.values())


def test_sample_domain_returns_domain():
    c = PerplexityCorrelationCurriculum(["a", "b"], freeze_until_tokens=0)
    rng = random.Random(0)
    d = c.sample_domain(rng)
    assert d in ("a", "b")
