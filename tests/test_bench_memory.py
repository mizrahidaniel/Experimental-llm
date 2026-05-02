"""Memory-scaling benchmark."""

from sprl.bench.memory_scaling import MemoryScalingConfig, benchmark_memory_scaling

from tests._eval_helpers import tiny_model


def test_memory_benchmark_runs():
    model = tiny_model()
    out = benchmark_memory_scaling(
        model, MemoryScalingConfig(batch_sizes=[1], seq_lens=[8], warmup=0)
    )
    assert "measurements" in out
    assert len(out["measurements"]) == 1
    cell = out["measurements"][0]
    assert cell["batch"] == 1
    assert cell["seq_len_patches"] == 8
    assert "rss_after_mb" in cell
    assert cell["rss_after_mb"] > 0
    assert out["device"] == "cpu"


def test_memory_benchmark_multi_cell():
    model = tiny_model()
    out = benchmark_memory_scaling(
        model,
        MemoryScalingConfig(batch_sizes=[1, 2], seq_lens=[8, 16], warmup=0),
    )
    assert len(out["measurements"]) == 4


def test_memory_benchmark_backward():
    model = tiny_model()
    out = benchmark_memory_scaling(
        model,
        MemoryScalingConfig(
            batch_sizes=[1], seq_lens=[8], include_backward=True, warmup=0
        ),
    )
    assert len(out["measurements"]) == 1
