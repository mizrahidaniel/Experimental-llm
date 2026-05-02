"""Inference utilities for SPRL-v2.

Components:
  - kv_cache: KVCache for MLA (compressed c_KV + decoupled rotary key K_R).
  - sampler: temperature / top-p / top-k byte sampler.
  - speculative: speculative decoding via the MTP head.
  - chat: byte-level chat-template scaffolding (system / user / assistant).
  - generate: end-to-end driver, with optional TTT-mode Kalman memory.
"""

from sprl.inference.kv_cache import KVCache, MLAKVCache  # noqa: F401
from sprl.inference.sampler import byte_sampler  # noqa: F401
from sprl.inference.speculative import speculative_decode  # noqa: F401
from sprl.inference.chat import (  # noqa: F401
    apply_chat_template,
    parse_chat_response,
    SYSTEM_OPEN,
    SYSTEM_CLOSE,
    USER_OPEN,
    USER_CLOSE,
    ASSISTANT_OPEN,
    ASSISTANT_CLOSE,
)
from sprl.inference.generate import generate  # noqa: F401
