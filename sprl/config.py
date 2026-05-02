"""Hierarchical config for SPRL-v2.

All knobs are dataclasses so they can be pickled, diffed, and round-tripped
through YAML. Defaults match the pilot 200M config from the spec.

Aggressive options (Bets A/B/C, NVFP4, MoE) are opt-in via flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class PatcherConfig:
    enabled: bool = True
    byte_lm_layers: int = 2
    byte_lm_dim: int = 256
    entropy_threshold: float = 1.5
    max_patch_bytes: int = 16
    min_patch_bytes: int = 1
    byte_encoder_layers: int = 4
    byte_encoder_dim: int = 384
    patch_dim: int = 1024  # == d_model

    # Average-patch-size invariant (asserted in tests).
    target_avg_patch_bytes: float = 6.0


@dataclass
class MLAConfig:
    d_c_latent: int = 512  # KV latent dim
    d_qhead: int = 128
    d_rope_decoupled: int = 32
    n_heads: int = 16
    rope_base: float = 500_000.0


@dataclass
class DSAConfig:
    enabled: bool = True
    indexer_type: str = "hierarchical_hisa"  # or "flat"
    top_k_blocks: int = 32
    top_k_tokens_per_block: int = 64
    block_size: int = 128
    # Switchable dense<->sparse per InfLLM-V2.
    dense_until_seqlen: int = 1024


@dataclass
class SlidingConfig:
    window_size: int = 1024


@dataclass
class TropicalConfig:
    """Bet B — tropical (max-plus) attention heads."""

    enabled: bool = False  # default off; opt-in via kill-experiment config
    fraction_of_heads: float = 0.25  # 1 in 4 heads
    beta_warmup_min: float = 1.0
    beta_warmup_max: float = 32.0
    warmup_fraction: float = 0.05  # of total training steps
    fp8_fallback_during_warmup: bool = True


@dataclass
class AttentionConfig:
    pattern: str = "alternating"  # "alternating" | "all_dsa" | "all_sliding"
    layer_types: List[str] = field(
        default_factory=lambda: ["dsa_mla", "sliding_window"]
    )
    mla: MLAConfig = field(default_factory=MLAConfig)
    dsa: DSAConfig = field(default_factory=DSAConfig)
    sliding: SlidingConfig = field(default_factory=SlidingConfig)
    tropical: TropicalConfig = field(default_factory=TropicalConfig)


@dataclass
class KalmanMemoryConfig:
    """Bet A — info-form Kalman belief memory."""

    enabled: bool = False
    rank: int = 8
    init_F_diag: float = 0.95
    eps_diag: float = 1.0e-4  # added to D each step for numerical stability
    chunk_size: int = 256  # for chunked associative scan


@dataclass
class ActiveInferenceRouterConfig:
    enabled: bool = False  # tied to KalmanMemoryConfig.enabled
    k_max: int = 8
    k_target: int = 4
    init_kappa: float = 1.0
    pi_lr_lambda: float = 1.0e-3  # PI controller LR for compute-budget Lagrangian


@dataclass
class RGFlowConfig:
    """Bet C — renormalization-group block-spin regularizer on depth."""

    enabled: bool = False
    rank_T_b: int = 32
    init_method: str = "marchenko_pastur"
    alpha_ramp_steps_frac: float = 0.1
    alpha_target: float = 1.0e-3
    diagnostic_log_every: int = 1000


@dataclass
class DRAMConfig:
    enabled: bool = True
    weight_tied: bool = True
    k_iterations_default: int = 4
    k_max: int = 8
    use_depth_attention: bool = True
    use_expert_attention: bool = False
    rg_flow: RGFlowConfig = field(default_factory=RGFlowConfig)
    router: ActiveInferenceRouterConfig = field(
        default_factory=ActiveInferenceRouterConfig
    )


@dataclass
class MoEConfig:
    enabled: bool = True
    num_routed_experts: int = 64
    num_shared_experts: int = 4
    active_per_token: int = 6
    expert_dim: int = 1408  # SwiGLU FFN inner dim per expert
    router: str = "top_k_softmax"
    balancing: str = "aux_loss_free"  # bias-update rule
    aux_loss_weight: float = 0.001  # small backup
    expert_grouping: str = "heterogeneous"  # 32 small + 32 large
    bias_update_lr: float = 1.0e-3


@dataclass
class MTPConfig:
    enabled: bool = True
    depth: int = 2  # predict next 2 tokens
    loss_weight: float = 0.3


@dataclass
class PrecisionConfig:
    forward_default: str = "bf16"  # "bf16" | "fp8" | "nvfp4"
    backward_default: str = "bf16"
    sensitive_modules: List[str] = field(
        default_factory=lambda: [
            "rmsnorm_scales",
            "kalman_memory_state",
            "tropical_softmax_warmup",
            "lm_head",
        ]
    )
    optimizer_type: str = "adamw"  # "adamw" | "adamw_8bit"
    use_galore: bool = False
    galore_rank: int = 128
    galore_targets: List[str] = field(
        default_factory=lambda: ["moe_expert_w1", "moe_expert_w2", "moe_expert_w3"]
    )
    master_weight_dtype: str = "bf16"


@dataclass
class TrainingConfig:
    seed: int = 1234
    batch_size: int = 32
    seq_len_patches: int = 1024  # patches per sequence (multiply by ~6 for bytes)
    lr: float = 3.0e-3
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    total_steps: int = 200_000
    grad_clip: float = 1.0

    # Distillation
    distill_enabled: bool = False
    distill_teacher: str = "llama3-70b"
    distill_topk_logits: int = 32
    distill_kl_weight_init: float = 0.7
    distill_kl_weight_final: float = 0.3
    distill_cutoff_tokens: int = 300_000_000_000  # 300B tokens

    # NCA pre-pre-training
    nca_pretrain_enabled: bool = True
    nca_num_tokens: int = 200_000_000
    nca_reset_embed_after: bool = True

    # Curriculum (perplexity-correlation reweighting)
    curriculum_enabled: bool = False
    curriculum_freeze_until_tokens: int = 50_000_000_000


@dataclass
class SPRLConfig:
    """Top-level config bundling every component."""

    # Core dims
    d_model: int = 1024
    n_layers: int = 24
    vocab_size: int = 256  # byte-level
    max_seq_len_patches: int = 8192

    # Sub-configs
    patcher: PatcherConfig = field(default_factory=PatcherConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    dram: DRAMConfig = field(default_factory=DRAMConfig)
    kalman: KalmanMemoryConfig = field(default_factory=KalmanMemoryConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "SPRLConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "SPRLConfig":
        """Recursively rebuild dataclasses from a YAML/dict.

        Uses the *default factory* of each field to discover sub-dataclass types
        (avoiding `f.type` which is a string under `from __future__ import annotations`).
        """
        import dataclasses as _dc

        def build(dc_cls, d):
            if d is None:
                return dc_cls()
            if not _dc.is_dataclass(dc_cls):
                return d
            kwargs = {}
            for f_ in _dc.fields(dc_cls):
                if f_.name not in d:
                    continue
                val = d[f_.name]
                # Resolve sub-dataclass type via the default_factory if present.
                sub_cls = None
                if f_.default_factory is not _dc.MISSING:
                    try:
                        sample = f_.default_factory()
                        if _dc.is_dataclass(sample):
                            sub_cls = type(sample)
                    except TypeError:
                        sub_cls = None
                if sub_cls is not None and isinstance(val, dict):
                    kwargs[f_.name] = build(sub_cls, val)
                else:
                    kwargs[f_.name] = val
            return dc_cls(**kwargs)

        return build(cls, raw)


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------


def default_pilot_200m() -> SPRLConfig:
    """200M-active pilot config (legacy v2 Stage 1)."""
    cfg = SPRLConfig(
        d_model=768,
        n_layers=12,
        max_seq_len_patches=2048,
    )
    cfg.attention.mla.n_heads = 12
    cfg.attention.mla.d_c_latent = 384
    cfg.moe.num_routed_experts = 16
    cfg.moe.num_shared_experts = 2
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 1024
    return cfg


def default_pilot_100m() -> SPRLConfig:
    """100M-active pilot for fast (~4-hour) Stage-1 kill experiments.

    Mirrors `configs/pilot_100m_bf16.yaml`. Bets A/B/C default off; turn on
    individually for kill_exp_{A,B,C}.
    """
    cfg = SPRLConfig(
        d_model=512,
        n_layers=12,
        max_seq_len_patches=1024,
    )
    cfg.patcher.byte_lm_dim = 192
    cfg.patcher.byte_encoder_layers = 2
    cfg.patcher.byte_encoder_dim = 256
    cfg.patcher.patch_dim = 512
    cfg.attention.mla.n_heads = 8
    cfg.attention.mla.d_c_latent = 256
    cfg.attention.mla.d_qhead = 64
    cfg.attention.mla.d_rope_decoupled = 16
    cfg.attention.dsa.top_k_blocks = 8
    cfg.dram.k_iterations_default = 3
    cfg.dram.k_max = 6
    cfg.dram.router.k_max = 6
    cfg.dram.router.k_target = 3
    cfg.moe.num_routed_experts = 8
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 1024
    cfg.training.batch_size = 32
    cfg.training.seq_len_patches = 512
    cfg.training.lr = 6.0e-3
    cfg.training.warmup_steps = 1000
    cfg.training.total_steps = 30000
    return cfg


def default_full_500m() -> SPRLConfig:
    """500M-active full-run config (~2B total), the realistic Stage-3 target.

    Mirrors `configs/full_500m_bf16.yaml`. All three bets default on; demote
    individually based on Stage-1 kill outcomes. No teacher distillation in
    the default plan (local training only); flip `training.distill_enabled`
    to use precomputed top-32 NPZ teacher logits.
    """
    cfg = SPRLConfig(
        d_model=1024,
        n_layers=16,
        max_seq_len_patches=8192,
    )
    cfg.attention.mla.n_heads = 16
    cfg.attention.mla.d_c_latent = 384
    cfg.attention.mla.d_qhead = 96
    cfg.attention.mla.d_rope_decoupled = 32
    cfg.attention.dsa.top_k_blocks = 16
    cfg.attention.tropical.enabled = True
    cfg.dram.k_iterations_default = 3
    cfg.dram.k_max = 6
    cfg.dram.use_depth_attention = True
    cfg.dram.rg_flow.enabled = True
    cfg.dram.router.enabled = True
    cfg.dram.router.k_max = 6
    cfg.dram.router.k_target = 3
    cfg.kalman.enabled = True
    cfg.moe.num_routed_experts = 32
    cfg.moe.num_shared_experts = 2
    cfg.moe.active_per_token = 4
    cfg.moe.expert_dim = 2048
    cfg.precision.optimizer_type = "adamw_8bit"
    cfg.precision.use_galore = True
    cfg.training.batch_size = 16
    cfg.training.seq_len_patches = 1024
    cfg.training.warmup_steps = 2000
    cfg.training.total_steps = 100_000
    cfg.training.distill_enabled = False
    cfg.training.nca_pretrain_enabled = True
    cfg.training.curriculum_enabled = True
    cfg.training.curriculum_freeze_until_tokens = 5_000_000_000
    return cfg


def kill_experiment_A() -> SPRLConfig:
    cfg = default_pilot_100m()
    cfg.kalman.enabled = True
    cfg.dram.router.enabled = True
    return cfg


def kill_experiment_B() -> SPRLConfig:
    cfg = default_pilot_100m()
    cfg.attention.tropical.enabled = True
    return cfg


def kill_experiment_C() -> SPRLConfig:
    cfg = default_pilot_100m()
    cfg.dram.rg_flow.enabled = True
    return cfg
