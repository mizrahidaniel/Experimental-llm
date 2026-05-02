"""Forward+backward throughput benchmark.

Returns tokens/sec where one "token" is one byte-slot in the byte LM head
(so a sequence of S patches × max_patch_bytes contributes S * B_p tokens).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from sprl.model import SPRLv2


@dataclass
class ThroughputConfig:
    batch: int = 2
    seq_len_patches: int = 32
    n_iters: int = 5
    warmup: int = 2
    include_backward: bool = True


def benchmark_throughput(
    model: SPRLv2, config: Optional[ThroughputConfig] = None
) -> Dict[str, float]:
    cfg = config or ThroughputConfig()
    device = next(model.parameters()).device
    max_pb = model.cfg.patcher.max_patch_bytes

    z = torch.randn(cfg.batch, cfg.seq_len_patches, model.cfg.d_model, device=device)
    targets = torch.randint(
        0,
        model.cfg.vocab_size,
        (cfg.batch, cfg.seq_len_patches, max_pb),
        device=device,
    )

    def _step():
        out = model.forward_from_patches(z)
        if cfg.include_backward:
            loss = F.cross_entropy(
                out.byte_logits.reshape(-1, model.cfg.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            for p in model.parameters():
                p.grad = None

    # Warmup.
    for _ in range(cfg.warmup):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(cfg.n_iters):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tokens_per_step = cfg.batch * cfg.seq_len_patches * max_pb
    total_tokens = tokens_per_step * cfg.n_iters
    tokens_per_sec = total_tokens / max(dt, 1e-9)
    return {
        "tokens_per_sec": tokens_per_sec,
        "step_time_ms": (dt / max(cfg.n_iters, 1)) * 1000,
        "total_seconds": dt,
        "n_iters": float(cfg.n_iters),
        "tokens_per_step": float(tokens_per_step),
        "device": 1.0 if device.type == "cuda" else 0.0,
    }
