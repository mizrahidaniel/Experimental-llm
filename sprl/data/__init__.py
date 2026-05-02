"""SPRL-v2 data pipeline.

Public API:

  build_data_iterator(cfg)             — top-level factory (loader.py)
  PatcherCollator                      — bytes → padded patches → batch dicts
  DomainMixer                          — interleave domain iterators
  iter_chunked_bytes / read_byte_files — low-level utilities

  Per-domain iterator factories:
    fineweb_edu_stream
    dclm_stream
    starcoder_v2_stream
    math_pile_stream
    synthetic_phi4_stream
    rstar_math_stream
    nca_trajectory_stream
    long_context_stream
"""

from __future__ import annotations

from sprl.data.bytes import (
    PatcherCollator,
    CollatorConfig,
    chunk_byte_stream,
    iter_chunked_bytes,
    read_byte_files,
)
from sprl.data.mixer import DomainMixer
from sprl.data.fineweb_edu import fineweb_edu_stream
from sprl.data.dclm import dclm_stream
from sprl.data.starcoder_v2 import starcoder_v2_stream
from sprl.data.math_pile import math_pile_stream
from sprl.data.synthetic_phi4 import synthetic_phi4_stream
from sprl.data.rstar_math import rstar_math_stream
from sprl.data.nca_trajectories import nca_trajectory_stream
from sprl.data.long_context import long_context_stream
from sprl.data.loader import build_data_iterator, DataConfig
from sprl.data.tokenized import (
    TokenizedShardDataset,
    TokenizedManifest,
    pack_batches,
    read_manifest,
    tokenized_iterator,
)

__all__ = [
    "PatcherCollator",
    "CollatorConfig",
    "chunk_byte_stream",
    "iter_chunked_bytes",
    "read_byte_files",
    "DomainMixer",
    "fineweb_edu_stream",
    "dclm_stream",
    "starcoder_v2_stream",
    "math_pile_stream",
    "synthetic_phi4_stream",
    "rstar_math_stream",
    "nca_trajectory_stream",
    "long_context_stream",
    "build_data_iterator",
    "DataConfig",
    "TokenizedShardDataset",
    "TokenizedManifest",
    "pack_batches",
    "read_manifest",
    "tokenized_iterator",
]
