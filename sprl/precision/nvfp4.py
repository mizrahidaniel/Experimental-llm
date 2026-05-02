"""NVFP4 emulation.

The real NVFP4 path goes through transformer_engine.pytorch / cublas FP4.
On Blackwell with TE ≥ 2.0, prefer that path. This module provides a *pure
PyTorch* emulation that:

  - Quantizes a tensor to E2M1 (4-bit, ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}) with a
    per-microblock FP8-E4M3 scale.
  - Round-trips back to BF16 for forward use.

It is correct (not fast). Use it as an oracle in unit tests; do NOT rely on
it for real training throughput.
"""

from __future__ import annotations

import torch
from torch import Tensor


# E2M1: 1 sign + 2 exponent bits (bias 1) + 1 mantissa bit
# Encoded values in magnitude:
_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _quantize_e2m1(x: Tensor) -> Tensor:
    """Round each element to the nearest E2M1 level (with sign)."""
    sign = torch.sign(x)
    mag = x.abs()
    # Find nearest level for each element via broadcasting.
    levels = _E2M1_LEVELS.to(x.device)
    # Compute |mag - levels| and pick argmin.
    d = (mag.unsqueeze(-1) - levels).abs()  # [..., 8]
    idx = d.argmin(dim=-1)
    out = sign * levels[idx]
    return out


def nvfp4_quantize_emulated(
    x: Tensor, microblock: int = 16
) -> tuple[Tensor, Tensor, Tensor]:
    """Emulate NVFP4: per-microblock-of-16 FP8 scale + per-tensor BF16 scale.

    Returns:
        x_int_levels: same shape as x, with values restricted to E2M1 levels (BF16).
        per_block_scale: [..., n_blocks] FP8 scale (here: float32, since CPU
            torch lacks an FP8 dtype on this platform).
        per_tensor_scale: scalar BF16 scale.
    """
    flat_shape = x.shape
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    pad = (microblock - n % microblock) % microblock
    if pad:
        x_flat = torch.cat([x_flat, x_flat.new_zeros(pad)])
    blocks = x_flat.view(-1, microblock)

    # Per-tensor BF16 scale: max-abs / 6.0 (E2M1 max magnitude).
    per_tensor = blocks.abs().amax().clamp_min(1.0e-12) / 6.0
    blocks_norm = blocks / per_tensor

    # Per-microblock FP8 scale: max-abs of normalized block / 6.0.
    per_block = blocks_norm.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / 6.0
    blocks_q = blocks_norm / per_block
    blocks_q = _quantize_e2m1(blocks_q)
    levels_recon = blocks_q * per_block * per_tensor
    levels_recon = levels_recon.reshape(-1)
    if pad:
        levels_recon = levels_recon[:-pad]
    return (
        levels_recon.reshape(flat_shape).to(x.dtype),
        per_block.reshape(-1),
        per_tensor,
    )


def nvfp4_dequantize_emulated(
    quantized: Tensor, per_block: Tensor, per_tensor: Tensor
) -> Tensor:
    """No-op since `nvfp4_quantize_emulated` already returns reconstructed values
    in `quantized`; provided for API symmetry."""
    return quantized


def nvfp4_roundtrip_emulated(x: Tensor) -> Tensor:
    """Quantize → reconstruct in one call. Useful for unit tests of the policy."""
    q, b, t = nvfp4_quantize_emulated(x)
    return q
