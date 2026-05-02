"""Byte-level sampler with temperature / top-p / top-k.

Deterministic when called with a fixed `torch.Generator`. Returns a single
byte id (Python int) per call.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def byte_sampler(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample one byte id from `logits`.

    Args:
      logits: 1-D tensor of shape [vocab].
      temperature: divides logits before softmax. 0 → argmax.
      top_p: nucleus threshold in (0, 1].
      top_k: if given, restrict to the top-k tokens.
      generator: torch.Generator for determinism.
    """
    assert logits.dim() == 1, "byte_sampler expects [vocab] logits"
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled = logits / max(temperature, 1.0e-6)

    # Top-k truncation.
    if top_k is not None and top_k > 0 and top_k < scaled.shape[-1]:
        vals, idx = torch.topk(scaled, k=top_k)
        mask = torch.full_like(scaled, float("-inf"))
        mask[idx] = vals
        scaled = mask

    # Top-p (nucleus) truncation.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens whose *previous* cumulative ≤ top_p, including the first
        # — this is the standard "shift-right" trick so we always keep the
        # smallest set whose cumulative prob ≥ top_p.
        shifted = torch.zeros_like(cum, dtype=torch.bool)
        shifted[..., 1:] = cum[..., :-1] <= top_p
        shifted[..., 0] = True
        keep_mask = shifted
        new_scaled = torch.full_like(scaled, float("-inf"))
        kept_idx = sorted_idx[keep_mask]
        kept_vals = sorted_logits[keep_mask]
        new_scaled[kept_idx] = kept_vals
        scaled = new_scaled

    probs = torch.softmax(scaled, dim=-1)
    sample = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sample.item())
