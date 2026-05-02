"""Dynamic Sparse Attention (DSA), DeepSeek-V3.2 style.

For each query token, a "lightning indexer" (a small MLP scoring q-k pairs)
produces top-k key indices; attention is computed only over those indices.

Reference implementation. The real DSA on a 5080 uses a fused Triton kernel
(see DeepSeek-V3 inference repo); here we use gather-based indexing so the
reference is correct on CPU and small GPUs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LightningIndexer(nn.Module):
    """Small MLP s(q_t, k_j) → R scoring each query/key pair.

    Trained jointly with the rest of the model. Supervised by the dense-attn
    rank in InfLLM-V2 style switchable mode (caller's responsibility).
    """

    def __init__(self, head_dim: int, hidden: int = 64):
        super().__init__()
        self.proj_q = nn.Linear(head_dim, hidden, bias=False)
        self.proj_k = nn.Linear(head_dim, hidden, bias=False)
        self.scale = hidden ** -0.5

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        """Q: [B, H, Sq, d]; K: [B, H, Sk, d] → scores [B, H, Sq, Sk]."""
        q = self.proj_q(Q)
        k = self.proj_k(K)
        return torch.matmul(q, k.transpose(-2, -1)) * self.scale


class DynamicSparseAttention(nn.Module):
    """DSA wrapper that consumes per-head Q,K,V from MLA and returns sparse
    softmax-attended outputs over a top-k key set.

    Switchable behavior (per InfLLM-V2): for short sequences (≤ dense_until_seqlen),
    runs dense attention so the indexer learns alongside dense; otherwise sparse.
    """

    def __init__(
        self,
        head_dim: int,
        top_k_tokens: int = 2048,
        dense_until_seqlen: int = 1024,
        causal: bool = True,
        indexer_hidden: int = 64,
    ):
        super().__init__()
        self.indexer = LightningIndexer(head_dim, indexer_hidden)
        self.top_k = top_k_tokens
        self.dense_until_seqlen = dense_until_seqlen
        self.causal = causal
        self.scale = head_dim ** -0.5

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Q,K: [B, H, S, d_qk]; V: [B, H, S, d_v]."""
        B, H, S, _ = Q.shape

        # Switchable: short sequences run dense.
        if S <= self.dense_until_seqlen:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            if self.causal:
                m = torch.triu(
                    torch.full((S, S), float("-inf"), device=Q.device), diagonal=1
                )
                scores = scores + m
            if attn_mask is not None:
                scores = scores + attn_mask
            return torch.matmul(torch.softmax(scores, dim=-1), V)

        # Sparse path: indexer chooses top-k keys per query.
        index_scores = self.indexer(Q, K)  # [B, H, S, S]
        if self.causal:
            m = torch.triu(
                torch.full((S, S), float("-inf"), device=Q.device), diagonal=1
            )
            index_scores = index_scores + m

        k_eff = min(self.top_k, S)
        topk = index_scores.topk(k_eff, dim=-1)  # values [B,H,S,k]; indices [B,H,S,k]
        topk_idx = topk.indices  # [B, H, S, k]
        topk_vals = topk.values  # for invalid mask: positions where indexer score is -inf

        # Gather K, V at top-k indices per query.
        d_qk = K.shape[-1]
        d_v = V.shape[-1]

        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d_qk)
        K_sel = torch.gather(
            K.unsqueeze(2).expand(-1, -1, S, -1, -1), dim=3, index=gather_idx
        )

        gather_idx_v = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d_v)
        V_sel = torch.gather(
            V.unsqueeze(2).expand(-1, -1, S, -1, -1), dim=3, index=gather_idx_v
        )

        # Compute exact attention scores (not indexer's) over the gathered subset.
        Q_e = Q.unsqueeze(-2)  # [B, H, S, 1, d_qk]
        scores = (Q_e * K_sel).sum(dim=-1) * self.scale  # [B, H, S, k]
        # Re-mask: if the indexer score was -inf (e.g. causal-masked future),
        # mask the final score too. This is the correct sparse-causal semantic.
        invalid = ~torch.isfinite(topk_vals)
        scores = scores.masked_fill(invalid, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # [B, H, S, k]
        out = (attn.unsqueeze(-1) * V_sel).sum(dim=-2)  # [B, H, S, d_v]
        return out
