"""Multi-head Latent Attention (MLA), DeepSeek-V3 style.

Compresses K and V into a low-rank latent c_KV ∈ ℝ^{d_c}, decompressed at
attention time. The KV cache stores only c_KV (and the decoupled rotary key),
not full per-head K/V.

Equation set:
  c_KV = W_DKV · h           # [..., seq, d_c]
  K_C  = W_UK  · c_KV        # [..., seq, n_h*d_h]
  V_C  = W_UV  · c_KV
  K_R  = RoPE(W_KR · h)      # [..., seq, n_h*d_r]
  Q_C  = W_DQ · h            # [..., seq, n_h*d_h]
  Q_R  = RoPE(W_QR · h)      # [..., seq, n_h*d_r]
  attn = softmax( [Q_C, Q_R] · [K_C, K_R]ᵀ / sqrt(d_h+d_r) ) · V_C
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sprl.attention.rope import build_rope_cache, apply_rope


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_c_latent: int,
        d_qhead: int,
        d_rope_decoupled: int,
        rope_base: float = 500_000.0,
        causal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_c = d_c_latent
        self.d_h = d_qhead
        self.d_r = d_rope_decoupled
        self.causal = causal

        # KV down/up.
        self.W_DKV = nn.Linear(d_model, d_c_latent, bias=False)
        self.W_UK = nn.Linear(d_c_latent, n_heads * d_qhead, bias=False)
        self.W_UV = nn.Linear(d_c_latent, n_heads * d_qhead, bias=False)
        # K decoupled rotary (single key per head — broadcast V style).
        self.W_KR = nn.Linear(d_model, n_heads * d_rope_decoupled, bias=False)

        # Q content + rotary.
        self.W_QC = nn.Linear(d_model, n_heads * d_qhead, bias=False)
        self.W_QR = nn.Linear(d_model, n_heads * d_rope_decoupled, bias=False)

        # Output projection.
        self.W_O = nn.Linear(n_heads * d_qhead, d_model, bias=False)

        self.scale = (d_qhead + d_rope_decoupled) ** -0.5
        self.rope_base = rope_base

    # ------------------------------------------------------------------
    def project_kv(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """h: [B, S, d_model] → (c_KV, K_R, V_C)."""
        c_KV = self.W_DKV(h)  # [B, S, d_c]
        K_R = self.W_KR(h)  # [B, S, n_h*d_r]
        V_C = self.W_UV(c_KV)  # [B, S, n_h*d_h]
        return c_KV, K_R, V_C

    def project_q(self, h: Tensor) -> tuple[Tensor, Tensor]:
        Q_C = self.W_QC(h)
        Q_R = self.W_QR(h)
        return Q_C, Q_R

    # ------------------------------------------------------------------
    def forward(
        self,
        h: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_attn_fn=None,
    ) -> Tensor:
        """h: [B, S, d_model] → [B, S, d_model].

        `head_attn_fn` lets you swap the per-head attention kernel, e.g. for
        DSA top-k routing or tropical attention. Signature:
            head_attn_fn(Q, K, V, causal_mask=None) -> [B, n_heads, S, d_h]
        where Q,K,V are [B, n_heads, S, d_h+d_r] (Q,K) and [B, n_heads, S, d_h] (V).
        """
        B, S, _ = h.shape
        c_KV, K_R, V_C = self.project_kv(h)
        Q_C, Q_R = self.project_q(h)

        # Up-project K from latent.
        K_C = self.W_UK(c_KV)  # [B, S, n_h*d_h]

        # Reshape to per-head.
        Q_C = Q_C.view(B, S, self.n_heads, self.d_h)
        Q_R = Q_R.view(B, S, self.n_heads, self.d_r)
        K_C = K_C.view(B, S, self.n_heads, self.d_h)
        K_R = K_R.view(B, S, self.n_heads, self.d_r)
        V_C = V_C.view(B, S, self.n_heads, self.d_h)

        # Apply RoPE to the rotary parts.
        cos, sin = build_rope_cache(S, self.d_r, base=self.rope_base, device=h.device, dtype=h.dtype)
        # [B, S, n_h, d_r] → apply per-head; use broadcasting with shape [1, S, 1, d_r/2]
        Q_R = apply_rope(Q_R.transpose(1, 2), cos, sin).transpose(1, 2)
        K_R = apply_rope(K_R.transpose(1, 2), cos, sin).transpose(1, 2)

        # Concatenate content + rotary on the head_dim axis.
        Q = torch.cat([Q_C, Q_R], dim=-1).transpose(1, 2)  # [B, n_h, S, d_h+d_r]
        K = torch.cat([K_C, K_R], dim=-1).transpose(1, 2)
        V = V_C.transpose(1, 2)  # [B, n_h, S, d_h]

        if head_attn_fn is None:
            out = self._dense_softmax_attn(Q, K, V, attn_mask=attn_mask)
        else:
            out = head_attn_fn(Q, K, V, attn_mask)

        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.d_h)
        return self.W_O(out)

    # ------------------------------------------------------------------
    def _dense_softmax_attn(
        self, Q: Tensor, K: Tensor, V: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Q,K: [B, H, S, d_h+d_r]; V: [B, H, S, d_h]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, S, S]
        if self.causal:
            S = scores.shape[-1]
            mask = torch.triu(
                torch.full((S, S), float("-inf"), device=scores.device), diagonal=1
            )
            scores = scores + mask
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)  # [B, H, S, d_h]
