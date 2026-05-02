"""Map module names to dtypes per the precision policy."""

from __future__ import annotations

from typing import Set

import torch

from sprl.config import PrecisionConfig


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8": torch.bfloat16,  # fallback when transformer_engine is missing
    "nvfp4": torch.bfloat16,  # ditto
    "fp32": torch.float32,
}


def select_dtype(cfg: PrecisionConfig, module_name: str, sensitive_keys: Set[str] | None = None) -> torch.dtype:
    keys = sensitive_keys or set(cfg.sensitive_modules)
    if module_name in keys:
        return _DTYPE_MAP.get(cfg.master_weight_dtype, torch.bfloat16)
    return _DTYPE_MAP.get(cfg.forward_default, torch.bfloat16)
