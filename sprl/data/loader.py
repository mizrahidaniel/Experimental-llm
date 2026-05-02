"""Top-level data-iterator factory.

`build_data_iterator(cfg)` returns an iterator of training batches shaped for
`SPRLv2.forward_from_patches`:

    {
      "patch_emb":      [B, S, d_model],
      "byte_targets":   [B, S, max_patch_bytes],
      "mtp_targets":    [B, S, depth],
      "attention_mask": [B, S],
      "lengths":        [B, S],
      "patch_bytes":    [B, S, max_patch_bytes],
    }

The caller is responsible for moving outputs to the training device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional

from sprl.config import SPRLConfig
from sprl.data.bytes import CollatorConfig, PatcherCollator
from sprl.data.dclm import dclm_stream
from sprl.data.fineweb_edu import fineweb_edu_stream
from sprl.data.long_context import long_context_stream
from sprl.data.math_pile import math_pile_stream
from sprl.data.mixer import DomainMixer
from sprl.data.rstar_math import rstar_math_stream
from sprl.data.starcoder_v2 import starcoder_v2_stream
from sprl.data.synthetic_phi4 import synthetic_phi4_stream
from sprl.patcher import ByteEncoder, ByteLM, EntropyPatcher
from sprl.training.curriculum import PerplexityCorrelationCurriculum


# Default mixture weights (sum to 1.0). Tuned to match the spec's pilot mixture
# (see §4.1): web text dominates, code+math each ~10%, synthetic + rstar small.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "fineweb_edu": 0.45,
    "dclm": 0.20,
    "starcoder_v2": 0.15,
    "math_pile": 0.10,
    "synthetic_phi4": 0.05,
    "rstar_math": 0.05,
}


SourceFactory = Callable[[], Iterator[bytes]]


@dataclass
class DataConfig:
    """Hyperparameters controlling `build_data_iterator`. Independent of
    `SPRLConfig` so callers can override without modifying the model config."""

    chunk_bytes: int = 512
    max_seq_patches: int = 64
    batch_size: int = 4
    use_hf: bool = False  # default off so unit tests don't hit the network
    long_context_enabled: bool = False
    long_context_seq_bytes: int = 64 * 1024
    weights: Optional[Dict[str, float]] = None
    domains: Optional[List[str]] = None  # subset of DEFAULT_WEIGHTS' keys
    use_curriculum: bool = False
    seed: int = 0
    max_examples_per_source: Optional[int] = None


def _default_factories(
    cfg: SPRLConfig,
    dcfg: DataConfig,
) -> Dict[str, SourceFactory]:
    seed = dcfg.seed
    use_hf = dcfg.use_hf
    max_ex = dcfg.max_examples_per_source

    factories: Dict[str, SourceFactory] = {
        "fineweb_edu": lambda: fineweb_edu_stream(
            seed=seed + 1, use_hf=use_hf, max_examples=max_ex
        ),
        "dclm": lambda: dclm_stream(
            seed=seed + 2, use_hf=use_hf, max_examples=max_ex
        ),
        "starcoder_v2": lambda: starcoder_v2_stream(
            seed=seed + 3, use_hf=use_hf, max_examples=max_ex
        ),
        "math_pile": lambda: math_pile_stream(
            seed=seed + 4, use_hf=use_hf, max_examples=max_ex
        ),
        "synthetic_phi4": lambda: synthetic_phi4_stream(
            seed=seed + 5, max_examples=max_ex, repeat=True
        ),
        "rstar_math": lambda: rstar_math_stream(
            seed=seed + 6, max_examples=max_ex, repeat=True
        ),
    }
    if dcfg.long_context_enabled:
        factories["long_context"] = lambda: long_context_stream(
            seed=seed + 7,
            seq_bytes=dcfg.long_context_seq_bytes,
            use_hf=use_hf,
            max_examples=max_ex,
        )
    return factories


def _build_collator(cfg: SPRLConfig, dcfg: DataConfig) -> PatcherCollator:
    pcfg = cfg.patcher
    byte_lm = ByteLM(dim=pcfg.byte_lm_dim, n_layers=pcfg.byte_lm_layers)
    patcher = EntropyPatcher(
        byte_lm,
        threshold=pcfg.entropy_threshold,
        min_patch_bytes=pcfg.min_patch_bytes,
        max_patch_bytes=pcfg.max_patch_bytes,
    )
    encoder = ByteEncoder(
        byte_dim=pcfg.byte_encoder_dim,
        patch_dim=cfg.d_model,
        n_layers=pcfg.byte_encoder_layers,
        max_patch_bytes=pcfg.max_patch_bytes,
    )
    coll_cfg = CollatorConfig(
        chunk_bytes=dcfg.chunk_bytes,
        max_seq_patches=dcfg.max_seq_patches,
        max_patch_bytes=pcfg.max_patch_bytes,
        batch_size=dcfg.batch_size,
        mtp_depth=cfg.mtp.depth if cfg.mtp.enabled else 1,
    )
    return PatcherCollator(patcher=patcher, byte_encoder=encoder, cfg=coll_cfg)


def build_data_iterator(
    cfg: SPRLConfig,
    *,
    data_cfg: Optional[DataConfig] = None,
    collator: Optional[PatcherCollator] = None,
) -> Iterator[dict]:
    """Returns an iterator yielding training-ready batch dicts.

    If `collator` is None, a fresh patcher + byte encoder are constructed from
    `cfg.patcher`. For real training, build a single shared collator and pass
    it in so the byte LM oracle is reused.
    """
    dcfg = data_cfg or DataConfig()
    factories = _default_factories(cfg, dcfg)

    if dcfg.domains is not None:
        factories = {k: v for k, v in factories.items() if k in dcfg.domains}

    if dcfg.weights is not None:
        weights = {k: dcfg.weights.get(k, 0.0) for k in factories}
    else:
        weights = {k: DEFAULT_WEIGHTS.get(k, 1.0 / len(factories)) for k in factories}

    curriculum = None
    if dcfg.use_curriculum:
        curriculum = PerplexityCorrelationCurriculum(
            domains=list(factories.keys()),
            freeze_until_tokens=cfg.training.curriculum_freeze_until_tokens,
        )
        s = sum(weights.get(k, 0.0) for k in factories) or 1.0
        for k in factories:
            curriculum.weights[k] = weights.get(k, 0.0) / s

    mixer = DomainMixer(
        sources=factories,
        curriculum=curriculum if dcfg.use_curriculum else None,
        weights=None if dcfg.use_curriculum else weights,
        seed=dcfg.seed,
    )

    coll = collator or _build_collator(cfg, dcfg)
    return coll.stream_batches(iter(mixer))
