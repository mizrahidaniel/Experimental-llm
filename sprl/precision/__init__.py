"""Precision policy.

This subpackage exposes the precision strategy referenced in Section 5 of the
spec. The default backend is BF16 fallback (works on any 5080 / Hopper / CPU).
NVFP4 / Quartet stochastic-rounding are stubs that defer to TransformerEngine
or Microxcaling at runtime.

Use `select_dtype(precision_cfg, module_name)` to pick the dtype for a given
module per the policy. Sensitive modules listed in the config stay in BF16
even when the global default is FP4.
"""

from sprl.precision.dtype_select import select_dtype  # noqa: F401
from sprl.precision.nvfp4 import (  # noqa: F401
    nvfp4_quantize_emulated,
    nvfp4_dequantize_emulated,
    nvfp4_roundtrip_emulated,
)
from sprl.precision.rht import random_hadamard  # noqa: F401
from sprl.precision.galore_optim import GaLoreProjector  # noqa: F401
