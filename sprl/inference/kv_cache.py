"""KV cache for MLA inference.

DeepSeek-V3 MLA stores per-layer the *compressed* latent c_KV (dim d_c) and
the decoupled rotary key K_R (dim n_heads × d_r); the per-head V_C is
recomputed from c_KV at attention time, eliminating the need to cache full
per-head K/V.

Simple ring buffer with append/truncate/clear, shape-typed for batch size 1
(the common inference path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class MLAKVCache:
    """Per-layer cache for one MLA module.

    Stores up to `max_len` tokens. `length` tracks the live prefix.
    """

    max_len: int
    d_c: int
    n_heads: int
    d_r: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32

    c_KV: Optional[Tensor] = None
    K_R: Optional[Tensor] = None
    length: int = 0

    def __post_init__(self):
        self.c_KV = torch.zeros(
            1, self.max_len, self.d_c, device=self.device, dtype=self.dtype
        )
        self.K_R = torch.zeros(
            1, self.max_len, self.n_heads * self.d_r, device=self.device, dtype=self.dtype
        )

    def append(self, c_KV_new: Tensor, K_R_new: Tensor) -> None:
        """Append a chunk of `S` tokens; raises if it would overflow."""
        assert c_KV_new.dim() == 3 and c_KV_new.shape[0] == 1
        S = c_KV_new.shape[1]
        if self.length + S > self.max_len:
            raise RuntimeError(
                f"MLA cache overflow: {self.length} + {S} > {self.max_len}"
            )
        self.c_KV[:, self.length : self.length + S] = c_KV_new
        self.K_R[:, self.length : self.length + S] = K_R_new
        self.length += S

    def get(self) -> tuple[Tensor, Tensor]:
        """Return the live (c_KV, K_R) prefix views."""
        return self.c_KV[:, : self.length], self.K_R[:, : self.length]

    def truncate(self, new_len: int) -> None:
        """Roll back the cache to `new_len` (e.g. for speculative reject)."""
        assert 0 <= new_len <= self.length
        self.length = new_len

    def clear(self) -> None:
        self.length = 0


@dataclass
class KVCache:
    """Multi-layer wrapper. Each layer has one MLAKVCache."""

    per_layer: List[MLAKVCache]

    @classmethod
    def for_model(
        cls,
        n_layers: int,
        max_len: int,
        d_c: int,
        n_heads: int,
        d_r: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "KVCache":
        dev = torch.device(device)
        return cls(
            per_layer=[
                MLAKVCache(
                    max_len=max_len,
                    d_c=d_c,
                    n_heads=n_heads,
                    d_r=d_r,
                    device=dev,
                    dtype=dtype,
                )
                for _ in range(n_layers)
            ]
        )

    @property
    def length(self) -> int:
        return self.per_layer[0].length if self.per_layer else 0

    def truncate(self, new_len: int) -> None:
        for layer in self.per_layer:
            layer.truncate(new_len)

    def clear(self) -> None:
        for layer in self.per_layer:
            layer.clear()
