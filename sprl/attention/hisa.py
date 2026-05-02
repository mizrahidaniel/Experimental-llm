"""Hierarchical Indexed Sparse Attention (HISA).

Replaces DSA's flat O(L²) indexer with a block→token cascade:
  1. Cluster keys into B = √L blocks (or a fixed block_size).
  2. For each query, score blocks → pick top-k_blocks blocks.
  3. Within selected blocks, score tokens → pick top-k_tokens.
Reduces indexer cost from O(L²) to ~O(L^1.5).

Per arXiv 2603.28458.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HierarchicalIndexer(nn.Module):
    def __init__(
        self,
        head_dim: int,
        block_size: int = 128,
        top_k_blocks: int = 32,
        top_k_tokens_per_block: int = 64,
        hidden: int = 64,
    ):
        super().__init__()
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.top_k_tokens_per_block = top_k_tokens_per_block

        self.proj_q_block = nn.Linear(head_dim, hidden, bias=False)
        self.proj_kblock = nn.Linear(head_dim, hidden, bias=False)
        self.proj_q_tok = nn.Linear(head_dim, hidden, bias=False)
        self.proj_k_tok = nn.Linear(head_dim, hidden, bias=False)
        self.scale = hidden ** -0.5

    @staticmethod
    def _block_pool(K: Tensor, block_size: int) -> tuple[Tensor, int]:
        """Mean-pool K within blocks of `block_size`. Pads to multiple of block_size.

        K: [B, H, S, d] → [B, H, n_blocks, d], n_blocks
        """
        B, H, S, d = K.shape
        n_blocks = (S + block_size - 1) // block_size
        pad = n_blocks * block_size - S
        if pad > 0:
            K = F.pad(K, (0, 0, 0, pad))
        K = K.view(B, H, n_blocks, block_size, d).mean(dim=3)
        return K, n_blocks

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        causal: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns sparse attention output [B, H, Sq, d_v]."""
        B, H, S, d_qk = Q.shape
        d_v = V.shape[-1]

        # Block-level scoring.
        K_blocks, n_blocks = self._block_pool(K, self.block_size)  # [B, H, n_blocks, d]
        q_b = self.proj_q_block(Q)
        k_b = self.proj_kblock(K_blocks)
        block_scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * self.scale
        # [B, H, S, n_blocks]

        if causal:
            # Block index of each query.
            q_block_idx = torch.arange(S, device=Q.device) // self.block_size  # [S]
            block_arange = torch.arange(n_blocks, device=Q.device)  # [n_blocks]
            block_mask = block_arange.unsqueeze(0) > q_block_idx.unsqueeze(1)  # [S, n_blocks]
            block_scores = block_scores.masked_fill(
                block_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        kb_eff = min(self.top_k_blocks, n_blocks)
        top_blocks = block_scores.topk(kb_eff, dim=-1).indices  # [B, H, S, kb]

        # For each query, score tokens within selected blocks.
        # Build per-query gather index of token ids (size kb * block_size per query).
        per_q_token_ids = (
            top_blocks.unsqueeze(-1) * self.block_size
            + torch.arange(self.block_size, device=Q.device)
        )  # [B, H, S, kb, block_size]
        per_q_token_ids = per_q_token_ids.view(B, H, S, kb_eff * self.block_size)
        # Mask out-of-range tokens.
        valid = per_q_token_ids < S

        # Gather K, V — broadcasting via expand on the seq axis.
        # We need K[..., per_q_token_ids, :] per query. Do it via gather.
        # Replace OOB with 0 so gather works; we'll mask later.
        clamped = per_q_token_ids.clamp(max=S - 1)
        gather_K = clamped.unsqueeze(-1).expand(-1, -1, -1, -1, d_qk)
        K_sel = torch.gather(K.unsqueeze(2).expand(-1, -1, S, -1, -1), 3, gather_K)
        gather_V = clamped.unsqueeze(-1).expand(-1, -1, -1, -1, d_v)
        V_sel = torch.gather(V.unsqueeze(2).expand(-1, -1, S, -1, -1), 3, gather_V)

        # Token-level scoring within selected blocks.
        q_t = self.proj_q_tok(Q).unsqueeze(-2)  # [B, H, S, 1, h]
        k_t = self.proj_k_tok(K_sel)  # [B, H, S, kb*bs, h]
        token_scores = (q_t * k_t).sum(dim=-1) * self.scale  # [B, H, S, kb*bs]
        token_scores = token_scores.masked_fill(~valid, float("-inf"))

        # Causal: drop tokens > query index.
        if causal:
            q_idx = torch.arange(S, device=Q.device).view(1, 1, S, 1)
            token_scores = token_scores.masked_fill(per_q_token_ids > q_idx, float("-inf"))

        kt_eff = min(self.top_k_tokens_per_block * kb_eff, token_scores.shape[-1])
        top_t = token_scores.topk(kt_eff, dim=-1)

        # Use *exact* QK scores (not indexer scores) for the final softmax.
        d_per_q = top_t.indices  # [B, H, S, kt]
        gather_K2 = d_per_q.unsqueeze(-1).expand(-1, -1, -1, -1, d_qk)
        K_final = torch.gather(K_sel, 3, gather_K2)
        gather_V2 = d_per_q.unsqueeze(-1).expand(-1, -1, -1, -1, d_v)
        V_final = torch.gather(V_sel, 3, gather_V2)

        # Score & softmax.
        scale = d_qk ** -0.5
        scores = (Q.unsqueeze(-2) * K_final).sum(-1) * scale  # [B, H, S, kt]
        # Mask invalid.
        valid_top = top_t.values > float("-inf") / 2
        scores = scores.masked_fill(~valid_top, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn.unsqueeze(-1) * V_final).sum(-2)
        return out
