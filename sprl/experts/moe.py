"""Fine-grained MoE FFN block (DeepSeek-V3 style).

- N_routed routed experts (small SwiGLU FFNs).
- N_shared experts (always active for every token).
- Top-k routing with softmax, balanced via the ALF bias-update rule.
- Optional heterogeneous grouping (some smaller, some larger) — implemented
  by varying expert_dim per expert.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sprl.experts.routing import TopKSoftmaxRouter
from sprl.experts.alf_balancer import AuxLossFreeBalancer


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


class FineGrainedMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_routed: int = 64,
        n_shared: int = 4,
        active_per_token: int = 6,
        expert_dim: int = 1408,
        heterogeneous: bool = True,
        balancer_lr: float = 1.0e-3,
        aux_loss_weight: float = 0.001,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_routed = n_routed
        self.k = active_per_token
        self.aux_loss_weight = aux_loss_weight

        if heterogeneous:
            # 50% small (1.0× expert_dim), 50% large (1.5× expert_dim).
            half = n_routed // 2
            dims = [expert_dim] * half + [int(expert_dim * 1.5)] * (n_routed - half)
        else:
            dims = [expert_dim] * n_routed

        self.routed: List[SwiGLUExpert] = nn.ModuleList(
            [SwiGLUExpert(d_model, h) for h in dims]
        )
        self.shared: List[SwiGLUExpert] = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_dim) for _ in range(n_shared)]
        )

        self.router = TopKSoftmaxRouter(d_model, n_routed, k=active_per_token)
        self.balancer = AuxLossFreeBalancer(n_routed, lr=balancer_lr)

        # Observability buffers (updated each forward).
        self.register_buffer("last_load", torch.zeros(n_routed), persistent=False)
        self._last_logits = None
        self._last_topk_idx = None

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """x: [B, S, d_model] → [B, S, d_model]."""
        B, S, d = x.shape
        x_flat = x.reshape(B * S, d)

        topk_idx, topk_w, full_logits = self.router(x_flat)
        N = x_flat.shape[0]
        # Routed pass: gather tokens per expert, run, scatter back.
        out = torch.zeros_like(x_flat)
        # Shared pass — always.
        for s in self.shared:
            out = out + s(x_flat) / max(len(self.shared), 1)

        for e_id, expert in enumerate(self.routed):
            mask = (topk_idx == e_id)  # [N, k]
            if not mask.any():
                continue
            tok_ids, slot_ids = mask.nonzero(as_tuple=True)
            tokens = x_flat[tok_ids]  # [m, d]
            y = expert(tokens)  # [m, d]
            w = topk_w[tok_ids, slot_ids].unsqueeze(-1)  # [m, 1]
            out.index_add_(0, tok_ids, y * w)

        # Bookkeeping for ALF balancer.
        with torch.no_grad():
            counts = torch.bincount(topk_idx.flatten(), minlength=self.n_routed).float()
            self.last_load.copy_(counts)
        self._last_logits = full_logits.detach()
        self._last_topk_idx = topk_idx.detach()

        return out.reshape(B, S, d)

    # ------------------------------------------------------------------
    def aux_loss(self) -> Tensor | None:
        if self._last_logits is None or self.aux_loss_weight <= 0:
            return None
        return self.aux_loss_weight * self.balancer.aux_loss(
            self._last_logits, self._last_topk_idx, self.n_routed
        )

    @torch.no_grad()
    def step_balancer(self) -> None:
        if self._last_topk_idx is None:
            return
        self.balancer.update(self.router.bias, self._last_topk_idx)

    @torch.no_grad()
    def max_load_ratio(self) -> float:
        mean = self.last_load.mean().clamp_min(1.0)
        return float((self.last_load.max() / mean).item())
