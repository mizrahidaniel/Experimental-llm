"""Random Hadamard Transform.

NVIDIA's NVFP4 recipe applies an RHT before quantization to suppress outliers.
This is a deterministic, fast transform that "spreads" energy across all
positions, giving better FP4 quantization SNR.

We use the Walsh-Hadamard recursion combined with a random ±1 sign vector and
a random permutation. Power-of-two block size only.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _hadamard_matrix(n: int, device, dtype) -> Tensor:
    """Walsh-Hadamard matrix of size n (n must be a power of 2). Normalized to ±1/√n."""
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def random_hadamard(x: Tensor, seed: int = 0) -> Tensor:
    """Apply RHT along the last dim. Pads to next power-of-two; truncates back."""
    *lead, d = x.shape
    n = 1
    while n < d:
        n *= 2
    if n != d:
        pad = x.new_zeros(*lead, n - d)
        x_p = torch.cat([x, pad], dim=-1)
    else:
        x_p = x

    g = torch.Generator(device=x.device).manual_seed(seed)
    sign = (torch.rand(n, device=x.device, generator=g) > 0.5).float() * 2.0 - 1.0
    perm = torch.randperm(n, device=x.device, generator=g)

    H = _hadamard_matrix(n, device=x.device, dtype=x.dtype)
    y = x_p[..., perm] * sign
    y = y @ H
    return y[..., :d]
