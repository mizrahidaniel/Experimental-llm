"""SPRL-v2 layer blocks.

A layer is a (attention, ffn) pair where:
  - attention is one of MLA, MLA+DSA, sliding-window — chosen by the
    alternating pattern in `attention.layer_types`.
  - ffn is either a vanilla SwiGLU FFN or a fine-grained MoE.
  - Optionally tropical heads are spliced into MLA via `head_attn_fn`.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sprl.attention import (
    DynamicSparseAttention,
    MultiHeadLatentAttention,
    SlidingWindowAttention,
    build_dual_algebra_attn,
)
from sprl.attention.tropical import TropicalAttentionHead
from sprl.config import AttentionConfig, MoEConfig
from sprl.experts import FineGrainedMoE


# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


# ---------------------------------------------------------------------------


class _AttentionDispatcher(nn.Module):
    """Wraps an MLA module + optional DSA + tropical heads as a single nn.Module
    with .forward(h) returning [B, S, d_model]."""

    def __init__(
        self,
        mla: MultiHeadLatentAttention,
        dsa: Optional[DynamicSparseAttention] = None,
        tropical: Optional[TropicalAttentionHead] = None,
    ):
        super().__init__()
        self.mla = mla
        self.dsa = dsa
        self.tropical = tropical

    def forward(self, h: Tensor) -> Tensor:
        # Compose head_attn_fn from DSA and/or tropical, prioritising tropical
        # (a head is either tropical or sparse-softmax; not both in this prototype).
        head_fn = None
        if self.tropical is not None:
            head_fn = self.tropical
        elif self.dsa is not None:
            head_fn = self.dsa
        return self.mla(h, head_attn_fn=head_fn)


class SPRLLayer(nn.Module):
    """A single layer pre/post norm, attention + ffn.

    NB: SPRL-v2's *outer* layers are tied via the DRAMBlock; each layer
    instance here corresponds to one position in the tied recursion stack.
    """

    def __init__(
        self,
        d_model: int,
        attn_module: nn.Module,
        ffn_module: nn.Module,
    ):
        super().__init__()
        self.norm_attn = nn.RMSNorm(d_model)
        self.attn = attn_module
        self.norm_ffn = nn.RMSNorm(d_model)
        self.ffn = ffn_module

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------


def make_attention(
    layer_type: str,
    cfg: AttentionConfig,
    d_model: int,
) -> nn.Module:
    if layer_type == "dsa_mla":
        mla = MultiHeadLatentAttention(
            d_model=d_model,
            n_heads=cfg.mla.n_heads,
            d_c_latent=cfg.mla.d_c_latent,
            d_qhead=cfg.mla.d_qhead,
            d_rope_decoupled=cfg.mla.d_rope_decoupled,
            rope_base=cfg.mla.rope_base,
        )
        dsa = None
        if cfg.dsa.enabled:
            head_dim = cfg.mla.d_qhead + cfg.mla.d_rope_decoupled
            dsa = DynamicSparseAttention(
                head_dim=head_dim,
                top_k_tokens=cfg.dsa.top_k_blocks * cfg.dsa.top_k_tokens_per_block,
                dense_until_seqlen=cfg.dsa.dense_until_seqlen,
            )
        tropical = None
        if cfg.tropical.enabled:
            tropical = TropicalAttentionHead(
                n_heads=cfg.mla.n_heads,
                fraction_tropical=cfg.tropical.fraction_of_heads,
                beta_init=cfg.tropical.beta_warmup_min,
            )
        return _AttentionDispatcher(mla, dsa=dsa, tropical=tropical)

    if layer_type == "sliding_window":
        return SlidingWindowAttention(
            d_model=d_model,
            n_heads=cfg.mla.n_heads,
            window_size=cfg.sliding.window_size,
        )

    raise ValueError(f"Unknown layer_type {layer_type}")


def make_ffn(
    moe_cfg: MoEConfig, d_model: int, force_dense: bool = False
) -> nn.Module:
    if force_dense or not moe_cfg.enabled:
        return SwiGLUFFN(d_model, hidden=moe_cfg.expert_dim)
    return FineGrainedMoE(
        d_model=d_model,
        n_routed=moe_cfg.num_routed_experts,
        n_shared=moe_cfg.num_shared_experts,
        active_per_token=moe_cfg.active_per_token,
        expert_dim=moe_cfg.expert_dim,
        heterogeneous=(moe_cfg.expert_grouping == "heterogeneous"),
        balancer_lr=moe_cfg.bias_update_lr,
        aux_loss_weight=moe_cfg.aux_loss_weight,
    )
