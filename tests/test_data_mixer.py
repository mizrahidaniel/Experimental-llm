"""DomainMixer determinism, weighting, and reentrancy tests."""

from __future__ import annotations

from collections import Counter

from sprl.data.mixer import DomainMixer
from sprl.training.curriculum import PerplexityCorrelationCurriculum


def _make_factories():
    def f_a():
        return iter([("a", i) for i in range(1000)])

    def f_b():
        return iter([("b", i) for i in range(1000)])

    def f_c():
        return iter([("c", i) for i in range(1000)])

    return {"a": f_a, "b": f_b, "c": f_c}


def test_mixer_determinism():
    factories = _make_factories()
    m1 = DomainMixer(factories, seed=0)
    m2 = DomainMixer(factories, seed=0)
    s1 = [next(iter(m1)) for _ in range(0)]  # placeholder
    out1 = []
    out2 = []
    it1 = iter(m1)
    it2 = iter(m2)
    for _ in range(100):
        out1.append(next(it1))
        out2.append(next(it2))
    assert out1 == out2


def test_mixer_different_seed_different_order():
    factories = _make_factories()
    it1 = iter(DomainMixer(factories, seed=0))
    it2 = iter(DomainMixer(factories, seed=1))
    out1 = [next(it1) for _ in range(100)]
    out2 = [next(it2) for _ in range(100)]
    assert out1 != out2


def test_mixer_respects_weights():
    factories = _make_factories()
    weights = {"a": 0.8, "b": 0.1, "c": 0.1}
    m = DomainMixer(factories, weights=weights, seed=42)
    counts = Counter(item[0] for item in (next(iter(m)),) if False)  # warmup
    counts = Counter()
    it = iter(m)
    for _ in range(2000):
        counts[next(it)[0]] += 1
    # 'a' should dominate.
    assert counts["a"] > counts["b"] + counts["c"]
    # 'b' and 'c' similar.
    assert abs(counts["b"] - counts["c"]) < 0.5 * counts["a"]


def test_mixer_uses_curriculum_weights_live():
    factories = _make_factories()
    curr = PerplexityCorrelationCurriculum(domains=["a", "b", "c"], freeze_until_tokens=0)
    curr.weights = {"a": 0.9, "b": 0.05, "c": 0.05}
    m = DomainMixer(factories, curriculum=curr, seed=7)
    counts = Counter()
    it = iter(m)
    for _ in range(2000):
        counts[next(it)[0]] += 1
    assert counts["a"] > counts["b"] * 5


def test_mixer_reentrant():
    factories = _make_factories()
    m = DomainMixer(factories, seed=0)
    pass1 = [item for item, _ in zip(iter(m), range(50))]
    pass2 = [item for item, _ in zip(iter(m), range(50))]
    # Both passes start fresh (because factories are fresh).
    assert pass1 == pass2


def test_mixer_handles_empty_source():
    def f_empty():
        return iter([])

    def f_nonempty():
        return iter([("ok", i) for i in range(50)])

    m = DomainMixer(
        {"x": f_empty, "y": f_nonempty},
        weights={"x": 0.5, "y": 0.5},
        seed=0,
        restart_exhausted=False,
    )
    out = [item for item, _ in zip(iter(m), range(40))]
    # All yielded items must come from 'y' (since 'x' is empty).
    assert all(item[0] == "ok" for item in out)


def test_mixer_zero_weight_skips_source():
    factories = _make_factories()
    m = DomainMixer(
        factories, weights={"a": 1.0, "b": 0.0, "c": 0.0}, seed=0
    )
    it = iter(m)
    for _ in range(50):
        item = next(it)
        assert item[0] == "a"
