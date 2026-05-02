"""Depth-Recurrent Attention Mixture (DRAM) block.

A single weight-tied transformer block iterated K times per token. At each
iteration k, the block consumes:
  - The current latent z_k.
  - Local attention (dispatched through the layer's attention module).
  - Depth-attention over previous iterations (DepthAttention).

The number of iterations K can be:
  - constant (k_iterations_default), or
  - per-token via the ActiveInferenceRouter / EntropyRouter.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from sprl.recurrent.depth_attention import DepthAttention


class DRAMBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        ffn_module: nn.Module,
        use_depth_attention: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.attn = attention_module
        self.ffn = ffn_module
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.use_depth_attention = use_depth_attention
        if use_depth_attention:
            self.depth_attn = DepthAttention(d_model)
            self.norm_da = nn.RMSNorm(d_model)

    def step(self, z: Tensor, history: List[Tensor]) -> Tensor:
        """One iteration. Pre-norm residual."""
        h = self.attn(self.norm1(z))
        z = z + h
        if self.use_depth_attention:
            d = self.depth_attn(self.norm_da(z), history)
            z = z + d
        z = z + self.ffn(self.norm2(z))
        return z

    def forward(
        self,
        z0: Tensor,
        n_iter: int = 1,
        record_history: bool = True,
    ) -> tuple[Tensor, List[Tensor]]:
        """Iterate the block n_iter times. Returns (z_final, [z_0, …, z_n_iter])."""
        history: List[Tensor] = [z0]
        z = z0
        for _ in range(n_iter):
            z = self.step(z, history if self.use_depth_attention else [])
            history.append(z)
        if not record_history:
            history = [z0, z]  # for memory savings
        return z, history


class TokenLevelDRAMBlock(DRAMBlock):
    """Per-token iteration count (MoR-style).

    Uses a mask: at iteration k, only update tokens whose K*(t) > k. Frozen
    tokens carry their last value forward. Practical for batched compute.
    """

    def forward_token_level(
        self,
        z0: Tensor,
        k_per_token: Tensor,
        record_history: bool = False,
        max_iter_cap: Optional[int] = None,
    ) -> tuple[Tensor, List[Tensor]]:
        """z0: [B, S, d]. k_per_token: [B, S] long (≥ 1) or float (STE).

        Tokens whose iteration count is reached freeze: their value is held
        constant for the remaining iterations. The mask is built from a hard
        comparison; for STE/float k_per_token we compare against `.detach()`.

        Args:
          max_iter_cap: optional upper bound on iterations (uses min with the
            observed max).
        """
        if k_per_token.dtype.is_floating_point:
            max_iter = int(k_per_token.detach().max().item())
        else:
            max_iter = int(k_per_token.max().item())
        if max_iter_cap is not None:
            max_iter = min(max_iter, int(max_iter_cap))
        max_iter = max(1, max_iter)

        z = z0
        history: List[Tensor] = [z0]
        for k in range(max_iter):
            mask = (k_per_token > k).to(z.dtype).unsqueeze(-1)  # [B, S, 1]
            z_new = self.step(
                z, history if self.use_depth_attention else []
            )
            z = z_new * mask + z * (1.0 - mask)
            if record_history or self.use_depth_attention:
                history.append(z)
        if not record_history:
            history = [z0, z]
        return z, history
