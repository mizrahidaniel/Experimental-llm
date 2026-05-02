"""Memory-scaling benchmark.

Measures peak resident memory for varying (batch, seq_len) configurations.
Uses `torch.cuda.max_memory_allocated()` when CUDA is available, else falls
back to RSS via `resource.getrusage`.

Forward-only by default; pass `include_backward=True` to measure peak during a
backward pass (used to validate gradient checkpointing claims).
"""

from __future__ import annotations

import gc
import resource
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sprl.model import SPRLv2


@dataclass
class MemoryScalingConfig:
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2])
    seq_lens: List[int] = field(default_factory=lambda: [16, 64])
    include_backward: bool = False
    warmup: int = 1


def _peak_bytes_cuda(device: torch.device) -> int:
    return int(torch.cuda.max_memory_allocated(device))


def _rss_bytes() -> int:
    """Return current process RSS in bytes (Linux/macOS)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS. Heuristic: if value is tiny
    # in "bytes" interpretation (<1MB) treat as KB.
    if r < 10_000_000:
        return int(r) * 1024
    return int(r)


def _measure_one(
    model: SPRLv2,
    batch: int,
    seq: int,
    include_backward: bool,
    device: torch.device,
) -> Dict[str, float]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_bytes()

    z = torch.randn(batch, seq, model.cfg.d_model, device=device)
    if include_backward:
        z.requires_grad_(True)
        out = model.forward_from_patches(z)
        loss = out.byte_logits.float().mean()
        loss.backward()
    else:
        with torch.no_grad():
            _ = model.forward_from_patches(z)

    rss_after = _rss_bytes()
    metrics: Dict[str, float] = {
        "batch": float(batch),
        "seq_len_patches": float(seq),
        "rss_delta_mb": (rss_after - rss_before) / 1024 / 1024,
        "rss_after_mb": rss_after / 1024 / 1024,
    }
    if device.type == "cuda":
        metrics["cuda_peak_mb"] = _peak_bytes_cuda(device) / 1024 / 1024
    return metrics


def benchmark_memory_scaling(
    model: SPRLv2, config: Optional[MemoryScalingConfig] = None
) -> Dict[str, List[Dict[str, float]]]:
    """Sweep batch/seq grid, return list of per-cell measurements.

    Output dict has key 'measurements' with one dict per (batch, seq) cell.
    """
    cfg = config or MemoryScalingConfig()
    device = next(model.parameters()).device

    measurements: List[Dict[str, float]] = []
    # Warmup ensures alloc patterns are steady-state.
    if cfg.warmup > 0 and cfg.batch_sizes and cfg.seq_lens:
        for _ in range(cfg.warmup):
            _measure_one(
                model,
                cfg.batch_sizes[0],
                cfg.seq_lens[0],
                include_backward=False,
                device=device,
            )

    for b in cfg.batch_sizes:
        for s in cfg.seq_lens:
            measurements.append(
                _measure_one(model, b, s, cfg.include_backward, device)
            )

    return {"measurements": measurements, "device": str(device)}
