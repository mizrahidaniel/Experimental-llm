"""Throughput benchmark."""

from sprl.bench.throughput import ThroughputConfig, benchmark_throughput

from tests._eval_helpers import tiny_model


def test_throughput_forward_only():
    model = tiny_model()
    m = benchmark_throughput(
        model,
        ThroughputConfig(
            batch=1, seq_len_patches=8, n_iters=2, warmup=1, include_backward=False
        ),
    )
    assert m["tokens_per_sec"] > 0
    assert m["step_time_ms"] > 0
    assert m["n_iters"] == 2
    assert m["tokens_per_step"] == 8 * model.cfg.patcher.max_patch_bytes


def test_throughput_with_backward():
    model = tiny_model()
    m = benchmark_throughput(
        model,
        ThroughputConfig(
            batch=1, seq_len_patches=8, n_iters=2, warmup=1, include_backward=True
        ),
    )
    assert m["tokens_per_sec"] > 0
    # Backward should be slower (or comparable for tiny model).
    assert m["step_time_ms"] > 0
