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


def estimate_memory_breakdown(
    model: SPRLv2,
    batch: int = 16,
    seq_len_patches: int = 1024,
    bytes_per_param: int = 2,  # BF16 default
    include_optimizer: bool = True,
    include_teacher_logits: bool = False,
    teacher_topk: int = 32,
) -> Dict[str, float]:
    """Static memory estimate broken down by component, in GB.

    Estimator (no allocation; can be called before training begins):
      - weights:     N_params × bytes_per_param
      - gradients:   N_params × bytes_per_param  (BF16 grad)
      - optimizer:   N_params × 2  (8-bit AdamW with state-pair) — set 0 if disabled
      - activations: 4 × batch × seq × d_model × n_layers × bytes_per_param
                     (gradient checkpointing every 2 layers ⇒ ÷2)
      - kv_cache:    batch × seq × n_heads × (d_qhead + d_rope) × 2 × n_dsa_layers
      - kalman_state: batch × seq × d_model × (1 + rank) × n_layers (if Bet A on)
      - teacher_logits: batch × seq × max_patch_bytes × teacher_topk × 6 bytes
                        (int32 idx + fp16 logp), if distillation on
    """
    cfg = model.cfg
    n_params = sum(p.numel() for p in model.parameters())
    weights_gb = n_params * bytes_per_param / 1024 ** 3
    gradients_gb = weights_gb
    # 8-bit AdamW + GaLore on MoE has roughly 2 bytes/param of state.
    optimizer_gb = (n_params * 2 / 1024 ** 3) if include_optimizer else 0.0

    # Activations (with gradient checkpointing every 2 layers).
    act_per_layer = 4 * batch * seq_len_patches * cfg.d_model * bytes_per_param
    activations_gb = act_per_layer * cfg.n_layers / 2 / 1024 ** 3

    # KV cache for the MLA latent + decoupled rotary.
    head_dim = cfg.attention.mla.d_qhead + cfg.attention.mla.d_rope_decoupled
    n_dsa = sum(1 for t in cfg.attention.layer_types if t == "dsa_mla") * (
        cfg.n_layers // max(len(cfg.attention.layer_types), 1)
    )
    kv_gb = (
        batch * seq_len_patches * cfg.attention.mla.n_heads * head_dim * 2 * n_dsa
        * bytes_per_param / 1024 ** 3
    )

    # Kalman state: (η, D, U) per token per layer.
    kalman_gb = 0.0
    if cfg.kalman.enabled:
        per_token = cfg.d_model * (2 + cfg.kalman.rank)  # η + D + U cols
        kalman_gb = (
            batch * seq_len_patches * per_token * cfg.n_layers
            * bytes_per_param / 1024 ** 3
        )

    # MoE dispatch buffers (rough): batch × seq × active_per_token × expert_dim.
    moe_gb = 0.0
    if cfg.moe.enabled:
        moe_gb = (
            batch * seq_len_patches * cfg.moe.active_per_token * cfg.moe.expert_dim
            * bytes_per_param * 2 / 1024 ** 3   # ×2 for in+out
        )

    # Teacher logits (offline distillation batch buffer).
    teacher_gb = 0.0
    if include_teacher_logits:
        teacher_gb = (
            batch * seq_len_patches * cfg.patcher.max_patch_bytes
            * teacher_topk * 6 / 1024 ** 3
        )

    workspace_gb = 1.0  # PyTorch + NCCL + fragmentation slack

    total_gb = (
        weights_gb + gradients_gb + optimizer_gb + activations_gb
        + kv_gb + kalman_gb + moe_gb + teacher_gb + workspace_gb
    )

    return {
        "weights_gb": weights_gb,
        "gradients_gb": gradients_gb,
        "optimizer_states_gb": optimizer_gb,
        "activations_gb": activations_gb,
        "kv_or_attention_cache_gb": kv_gb,
        "kalman_state_gb": kalman_gb,
        "moe_buffers_gb": moe_gb,
        "teacher_logits_gb": teacher_gb,
        "workspace_gb": workspace_gb,
        "estimated_total_gb": total_gb,
        "n_params": float(n_params),
    }


def estimate_compute(
    model: SPRLv2,
    seq_len_patches: int = 1024,
    mean_recurrent_iterations: Optional[float] = None,
) -> Dict[str, float]:
    """Per-token compute accounting (v3.1, 6N convention with overhead factor).

    `flops_per_token = 6 × active_params × architecture_overhead_factor`
        (Chinchilla 6N: forward + backward).

    For middle-cell architectures, effective active per token is split:
        effective_active = prefix_active + cell_active × mean_k + suffix_active

    We approximate by counting MoE active vs inactive params and applying the
    K-multiplier only to params that live in the recurrent cell (when the
    middle-cell layout is in use). For uniform layers we treat the whole
    backbone as recurrent.
    """
    cfg = model.cfg
    if mean_recurrent_iterations is None:
        mean_recurrent_iterations = float(cfg.dram.k_iterations_default)

    n_total = sum(p.numel() for p in model.parameters())
    if cfg.moe.enabled:
        per_expert = 3 * cfg.d_model * cfg.moe.expert_dim
        moe_total = cfg.moe.num_routed_experts * per_expert * cfg.n_layers
        moe_active = cfg.moe.active_per_token * per_expert * cfg.n_layers
        active_params = n_total - (moe_total - moe_active)
    else:
        active_params = n_total

    flops_per_token_one_step = 6 * active_params

    # Effective FLOPs/token.
    middle_cell = cfg.dram.cell_type == "recurrent_middle_block"
    if middle_cell:
        prefix = max(cfg.dram.prefix_layers, 0)
        cell = max(cfg.dram.recurrent_cell_layers, 0)
        suffix = max(cfg.dram.suffix_layers, 0)
        total_layer_eq = prefix + cell + suffix
        # If specified shape is degenerate (e.g. zeros), fall back to uniform.
        if total_layer_eq <= 0:
            middle_cell = False

    if middle_cell:
        prefix = cfg.dram.prefix_layers
        cell = cfg.dram.recurrent_cell_layers
        suffix = cfg.dram.suffix_layers
        # Approximate per-layer active by averaging.
        per_layer_active = active_params / max(prefix + cell + suffix, 1)
        prefix_active = prefix * per_layer_active
        cell_active = cell * per_layer_active
        suffix_active = suffix * per_layer_active
        effective_active = prefix_active + cell_active * mean_recurrent_iterations + suffix_active
    else:
        prefix_active = 0.0
        cell_active = active_params
        suffix_active = 0.0
        if cfg.training.count_recurrence_in_active_flops:
            effective_active = active_params * mean_recurrent_iterations
        else:
            effective_active = active_params

    overhead = float(cfg.compute.architecture_overhead_factor)
    effective_flops = 6 * effective_active * overhead

    return {
        "n_params_total": float(n_total),
        "active_params_per_token": float(active_params),
        "active_params_per_recurrent_step": float(active_params),
        "mean_recurrent_iterations": float(mean_recurrent_iterations),
        "flops_per_token_one_step": float(flops_per_token_one_step),
        "effective_active_params_per_token": float(effective_active),
        "effective_flops_per_token": float(effective_flops),
        "architecture_overhead_factor": overhead,
        "prefix_active_params": float(prefix_active),
        "cell_active_params": float(cell_active),
        "suffix_active_params": float(suffix_active),
        "tokens_per_seq": float(seq_len_patches),
    }


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
