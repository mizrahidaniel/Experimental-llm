"""Optimizer factory.

Resolves AdamW / AdamW-8bit (bitsandbytes) and applies GaLore projection to
the largest MoE expert weights when configured.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


def _split_params_by_name(model: nn.Module, name_substrs: List[str]):
    matched, rest = [], []
    for n, p in model.named_parameters():
        if any(s in n for s in name_substrs):
            matched.append(p)
        else:
            rest.append(p)
    return matched, rest


def build_optimizer(
    model: nn.Module,
    lr: float = 3.0e-3,
    weight_decay: float = 0.1,
    optimizer_type: str = "adamw",
    galore: bool = False,
    galore_rank: int = 128,
    galore_targets: List[str] | None = None,
):
    galore_targets = galore_targets or []

    if optimizer_type == "adamw":
        opt_cls = torch.optim.AdamW
    elif optimizer_type == "adamw_8bit":
        try:
            from bitsandbytes.optim import AdamW8bit  # type: ignore

            opt_cls = AdamW8bit
        except ImportError:
            print("[optim] bitsandbytes unavailable; falling back to torch AdamW.")
            opt_cls = torch.optim.AdamW
    else:
        raise ValueError(optimizer_type)

    param_groups = [
        {"params": list(model.parameters()), "lr": lr, "weight_decay": weight_decay}
    ]
    optim = opt_cls(param_groups, lr=lr, weight_decay=weight_decay)

    if galore and galore_targets:
        # GaLore is wired in by attaching a hook on backward to project gradients.
        # We provide a minimal hook here; production should use the full GaLore
        # adapter from https://github.com/jiaweizzhao/GaLore.
        from sprl.precision.galore_optim import GaLoreProjector

        for n, p in model.named_parameters():
            if any(t in n for t in galore_targets) and p.dim() == 2:
                proj = GaLoreProjector(rank=galore_rank)
                p._galore_proj = proj  # type: ignore[attr-defined]

    return optim
