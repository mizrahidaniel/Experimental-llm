"""BLT entropy patcher.

Given a byte stream and a frozen byte-LM oracle, segment the stream at
high-entropy positions to produce variable-length patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn

from sprl.patcher.byte_lm import ByteLM


@dataclass
class PatchPlan:
    """Result of patching a single byte stream."""

    boundaries: List[int]  # positions where each patch *ends* (exclusive)
    starts: List[int]
    ends: List[int]

    @property
    def n_patches(self) -> int:
        return len(self.starts)

    @property
    def avg_len(self) -> float:
        if not self.starts:
            return 0.0
        return sum(e - s for s, e in zip(self.starts, self.ends)) / len(self.starts)


def patch_bytes_with_entropy(
    bytes_seq: Tensor,
    entropies: Tensor,
    threshold: float = 1.5,
    min_patch_bytes: int = 1,
    max_patch_bytes: int = 16,
) -> PatchPlan:
    """Greedy entropy-based patching.

    Walk the byte stream; close a patch whenever:
      (a) entropy at the current position > threshold and patch length ≥ min, or
      (b) patch length == max_patch_bytes (force-close).

    Args:
      bytes_seq: [T] int tensor of bytes.
      entropies: [T] float tensor — entropy of next-byte prediction at position t.
        Aligned so entropies[t] is "uncertainty about b_{t+1} given b_{≤t}".
      threshold: nats. Higher = larger patches.
      min_patch_bytes / max_patch_bytes: hard bounds.
    """
    assert bytes_seq.dim() == 1
    assert entropies.shape == bytes_seq.shape
    T = bytes_seq.shape[0]

    starts: List[int] = []
    ends: List[int] = []
    s = 0
    while s < T:
        # Find next boundary.
        e = s + min_patch_bytes
        e = min(e, T)
        max_e = min(s + max_patch_bytes, T)
        # Greedy walk.
        while e < max_e:
            # Entropies at position e-1 = "uncertainty about byte e".
            if float(entropies[e - 1]) > threshold:
                break
            e += 1
        starts.append(s)
        ends.append(e)
        s = e

    return PatchPlan(boundaries=ends.copy(), starts=starts, ends=ends)


class EntropyPatcher(nn.Module):
    """High-level wrapper: byte_lm + greedy patcher.

    Frozen — gradients should not flow into the byte_lm. Use `requires_grad_(False)`
    on instantiation.
    """

    def __init__(
        self,
        byte_lm: ByteLM,
        threshold: float = 1.5,
        min_patch_bytes: int = 1,
        max_patch_bytes: int = 16,
        freeze_byte_lm: bool = True,
    ):
        super().__init__()
        self.byte_lm = byte_lm
        self.threshold = threshold
        self.min_patch_bytes = min_patch_bytes
        self.max_patch_bytes = max_patch_bytes
        if freeze_byte_lm:
            for p in self.byte_lm.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def plan(self, bytes_seq: Tensor) -> List[PatchPlan]:
        """Plan patches for a [batch, T] byte tensor. Returns a list of PatchPlans."""
        if bytes_seq.dim() == 1:
            bytes_seq = bytes_seq.unsqueeze(0)
        H = self.byte_lm.next_byte_entropy(bytes_seq)  # [B, T]
        out = []
        for b in range(bytes_seq.shape[0]):
            out.append(
                patch_bytes_with_entropy(
                    bytes_seq[b],
                    H[b],
                    threshold=self.threshold,
                    min_patch_bytes=self.min_patch_bytes,
                    max_patch_bytes=self.max_patch_bytes,
                )
            )
        return out
