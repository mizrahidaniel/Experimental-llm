"""DistillScheduler: cutoff respected; weights anneal monotonically."""

from __future__ import annotations

from sprl.training.distill import DistillScheduler


def test_cutoff_zero_disables_distillation():
    s = DistillScheduler(init_w=0.7, final_w=0.3, anneal_tokens=10, cutoff_tokens=0)
    for t in (0, 1, 100, 10**12):
        assert s.weight(t) == 0.0


def test_anneal_monotonic_init_to_final():
    s = DistillScheduler(init_w=0.7, final_w=0.3, anneal_tokens=1000, cutoff_tokens=10_000)
    ws = [s.weight(t) for t in range(0, 1001, 100)]
    # Decreasing toward final_w.
    assert ws[0] == 0.7
    assert abs(ws[-1] - 0.3) < 1e-9
    for a, b in zip(ws, ws[1:]):
        assert a >= b - 1e-12


def test_post_anneal_holds_final_until_cutoff():
    s = DistillScheduler(init_w=0.7, final_w=0.3, anneal_tokens=1000, cutoff_tokens=5000)
    for t in (1000, 1500, 2000, 4999):
        assert abs(s.weight(t) - 0.3) < 1e-9
    assert s.weight(5000) == 0.0
    assert s.weight(10**11) == 0.0


def test_init_weight_at_zero_tokens():
    s = DistillScheduler(init_w=0.7, final_w=0.3, anneal_tokens=10, cutoff_tokens=20)
    assert abs(s.weight(0) - 0.7) < 1e-9
