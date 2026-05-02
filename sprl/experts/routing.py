"""Top-k softmax expert router.

Returns:
  topk_indices  [B*S, k]
  topk_weights  [B*S, k]   (re-normalized to sum 1 per token)
  raw_logits    [B*S, n_experts]   (used by the ALF bias rule)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TopKSoftmaxRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, k: int = 6):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_experts), requires_grad=False)
        self.k = k
        self.n_experts = n_experts

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x: [B, S, d] or [N, d]. Returns (topk_idx, topk_w, full_logits)."""
        if x.dim() == 3:
            B, S, d = x.shape
            x = x.reshape(B * S, d)
        logits = self.gate(x) + self.bias  # [N, E]
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)
        topk_w = F.softmax(topk_vals, dim=-1)  # re-normalize among chosen
        return topk_idx, topk_w, logits
