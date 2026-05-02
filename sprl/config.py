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
class TokenizerConfig:
    """v3.1 tokenizer selector.

    type:
      - "blt"  (legacy): byte-level entropy patcher; vocab_size=256.
      - "bpe"  (v3.1):   teacher-aligned BPE; vocab from `source` (Llama-3.1
                          → 128K, Qwen-2.5 → 152K, tiny_bpe → 32K offline).

    For `recommended` variant (v3.1) `type` MUST be "bpe" so direct
    teacher-token KL distillation is coherent. For `original_bets_mini` the
    default is "blt" + auxiliary_teacher_token_head.
    """

    type: str = "blt"  # "blt" | "bpe"
    source: str = "tiny_bpe"  # only used when type=="bpe"; matches teacher when distilling
    allow_network: bool = True  # set False in tests / sandboxes
    weight_tied: bool = True  # tie input embedding ↔ LM head (saves ~98M @ 128K vocab)


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
    """Tropical (max-plus) attention heads.

    In v3.1, tropical is NOT in the default `recommended` variant. It lives in
    `recommended_plus_tropical_probe` (one head, head-specific kill criterion)
    and in `original_bets_mini` (1 of ≤ 8 heads).
    """

    enabled: bool = False  # default off; opt-in via probe / original-bets variant
    fraction_of_heads: float = 0.25  # 1 in 4 heads
    beta_warmup_min: float = 1.0
    beta_warmup_max: float = 32.0
    warmup_fraction: float = 0.05  # of total training steps
    fp8_fallback_during_warmup: bool = True
    # v3.1: when False, the tropical head is built but its outputs are
    # discarded during training (eval-only probe). Currently treated equivalent
    # to `enabled=False` in the forward path; kept for config-level signalling.
    affects_training: bool = True
    # v3.1: original-bets requires ≤ 1/8 of heads for stability at mini scale.
    num_tropical_heads_total: int = 0  # 0 ⇒ derive from fraction_of_heads
    never_exceed_fraction: float = 0.25


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

    # v3.1: passive-probe gated residual fusion. When `mode == "passive_probe"`
    # the Kalman output (mean estimate) is fused into the hidden stream as
    #     h_fused = h + sigmoid(fusion_gate) · kalman_out
    # so the gate gets LM gradients. `fusion_gate_init = -3.0` ⇒ initial influence ≈ 0.05.
    # Use `mode == "diagnostic_probe"` to skip fusion entirely.
    mode: str = "passive_probe"  # "passive_probe" | "diagnostic_probe" | "active_if_kill_passes" | "passive_probe_then_active_if_passes"
    fusion_gate_init: float = -3.0
    can_control_router: bool = False  # gated on Bet A kill pass


@dataclass
class ActiveInferenceRouterConfig:
    enabled: bool = False  # tied to KalmanMemoryConfig.enabled
    k_max: int = 8
    k_target: int = 4
    init_kappa: float = 1.0
    pi_lr_lambda: float = 1.0e-3  # PI controller LR for compute-budget Lagrangian
    # v3.1: training mode. "straight_through_soft_k" makes routing fully
    # differentiable via STE. "controller_only" updates κ via a non-gradient
    # PI controller and trains the scout head supervised on oracle surprise.
    training_mode: str = "straight_through_soft_k"
    # Source of the epistemic signal:
    #   "scout_head_only"            — fallback when Kalman is passive-only
    #   "kalman_plus_scout_if_authority_else_scout_only"  — promote when authority granted
    source: str = "scout_head_only"


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

    # v3.1 middle-cell architecture.
    #
    # cell_type:
    #   "uniform_layers"           — legacy: every layer iterated K times.
    #   "recurrent_middle_block"   — v3.1: prefix(N) + tied-cell(M)·K + suffix(N).
    #
    # routing_granularity:
    #   "per_token"                — legacy: K* per token (incoherent w/ attention).
    #   "per_block"                — v3.1: K* per batch element (= "block").
    cell_type: str = "uniform_layers"
    prefix_layers: int = 0           # used only when cell_type=recurrent_middle_block
    recurrent_cell_layers: int = 0   # ditto; the *tied* cell depth
    suffix_layers: int = 0           # ditto
    routing_granularity: str = "per_token"


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
class ComputeConfig:
    """v3.1 compute / FLOP accounting.

    `flop_convention = 6N_per_token` is the Chinchilla forward+backward total
    cost. `architecture_overhead_factor` multiplies the bare 6N by a measured
    overhead (MoE routing, sparse-attn indexer, recurrence boilerplate, MTP
    head, etc.). Default 1.3; update from measured throughput after the first
    1B tokens.
    """

    flop_convention: str = "6N_per_token"
    architecture_overhead_factor: float = 1.3
    count_recurrence_in_active_params: bool = True
    target_effective_active_params: tuple = (250_000_000, 350_000_000)


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
    # Distillation modes:
    #   "teacher_token_kl"               — direct KL; requires identical tokenizers
    #   "auxiliary_teacher_token_head"   — KL on a per-trunk aux head in teacher vocab
    #   "sequence_level"                 — Kim & Rush 2016-style, student CE on teacher's argmax
    distill_mode: str = "teacher_token_kl"
    # v3.1: name the KL direction explicitly.
    #   "teacher_forward_kl"  — KL(teacher || student); standard distillation
    #   "reverse_kl"          — KL(student || teacher); MiniLLM-style mode-seeking
    distill_kl_direction: str = "teacher_forward_kl"
    # Phase-aware teachers per spec §6.4: base for raw web (Phase 3a), instruct
    # for synthetic / instruction (Phase 3c).
    distill_teacher_base: str = "llama_3_1_8b"
    distill_teacher_instruct: str = "llama_3_1_8b_instruct"
    # v3.1: top-K logit storage layout (set when distill_mode=teacher_token_kl).
    distill_store_indices: bool = True
    distill_store_logits: bool = True
    distill_store_teacher_logsumexp: bool = True
    distill_store_tail_mass: bool = False
    distill_index_dtype: str = "int32"  # 128K-152K teacher vocabs need >uint16

    # Active selection (Bet C — teacher-disagreement curriculum, default off).
    # Requires distill_enabled=True; selects training docs by student↔teacher KL
    # with a diversity regularizer to prevent collapse onto noisy text.
    active_selection_enabled: bool = False
    active_selection_pool_multiplier: int = 4  # draw 4× the desired batch, keep top 1×
    active_selection_diversity_lambda: float = 0.5
    active_selection_diversity_history: int = 1024
    # Iterative re-distillation pass after the main distillation phase.
    iterative_redistill_enabled: bool = False
    iterative_redistill_loss_gap_threshold: float = 1.2  # student/teacher loss ratio
    iterative_redistill_passes: int = 1

    # NCA pre-pre-training
    nca_pretrain_enabled: bool = True
    nca_num_tokens: int = 200_000_000
    nca_reset_embed_after: bool = True

    # Curriculum (perplexity-correlation reweighting)
    curriculum_enabled: bool = False
    curriculum_freeze_until_tokens: int = 50_000_000_000

    # Compute accounting. Setting `count_recurrence_in_active_flops=True` makes
    # `effective_flops_per_token = active_params × mean_recurrent_iterations`
    # in throughput / memory reports. False = report just `active_params` (the
    # standard convention for comparison with published baselines).
    count_recurrence_in_active_flops: bool = True


@dataclass
class SPRLConfig:
    """Top-level config bundling every component."""

    # Variant tag (v3.1); informs validation. Legacy configs that don't set
    # this default to "legacy" (validation skips the v3.1-specific checks).
    variant: str = "legacy"

    # Core dims
    d_model: int = 1024
    n_layers: int = 24
    vocab_size: int = 256  # byte-level
    max_seq_len_patches: int = 8192

    # Sub-configs
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    patcher: PatcherConfig = field(default_factory=PatcherConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    dram: DRAMConfig = field(default_factory=DRAMConfig)
    kalman: KalmanMemoryConfig = field(default_factory=KalmanMemoryConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

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
        cfg = cls.from_dict(raw)
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Hard checks for incoherent flag combinations.

        Raises ValueError on configurations that would silently produce
        meaningless training. Called by `from_yaml` / `from_dict`; callers
        constructing `SPRLConfig` directly can call it explicitly.
        """
        # BLT byte-level student vs. BPE-vocab teacher: a direct teacher-token
        # KL between a 256-class byte distribution and a 32K+ teacher vocab is
        # nonsense. The mode flag is a future-proofing for when alternative
        # alignment paths are wired in.
        if (
            self.patcher.enabled
            and self.training.distill_enabled
            and self.training.distill_mode == "teacher_token_kl"
        ):
            raise ValueError(
                "Cannot use direct teacher-token KL with BLT byte/patch logits "
                f"(student vocab={self.vocab_size}, teacher vocab is generally "
                "32K+ BPE). Set training.distill_mode to 'sequence_level' or "
                "'auxiliary_teacher_token_head', or disable BLT (patcher.enabled=false)."
            )

        # Recurrence is on but compute accounting ignores it: throughput / FLOP
        # reports will undercount by ≈ mean_k×.
        if (
            self.dram.enabled
            and self.dram.k_iterations_default > 1
            and not self.training.count_recurrence_in_active_flops
        ):
            raise ValueError(
                "Recurrent depth is enabled (k_iterations_default="
                f"{self.dram.k_iterations_default}) but compute accounting "
                "ignores recurrence. Set training.count_recurrence_in_active_flops=true "
                "or set dram.k_iterations_default=1."
            )

        # Active selection requires distillation to score against.
        if self.training.active_selection_enabled and not self.training.distill_enabled:
            raise ValueError(
                "training.active_selection_enabled=true requires "
                "training.distill_enabled=true (selection scores docs by "
                "student↔teacher KL)."
            )

        # Iterative re-distillation likewise.
        if self.training.iterative_redistill_enabled and not self.training.distill_enabled:
            raise ValueError(
                "training.iterative_redistill_enabled=true requires "
                "training.distill_enabled=true."
            )

        # ---- v3.1 hard checks --------------------------------------------------
        v3_variants = {
            "recommended", "recommended_plus_tropical_probe",
            "original_bets_mini_v3_1",
        }
        is_v3 = self.variant in v3_variants

        # (4) Recurrence cell-type — v3.1 mandates the middle-cell architecture.
        if is_v3 and self.dram.enabled and self.dram.cell_type != "recurrent_middle_block":
            raise ValueError(
                f"variant={self.variant} mandates dram.cell_type=recurrent_middle_block; "
                f"got {self.dram.cell_type!r}."
            )

        # (4b) Routing granularity — v3.1 mandates per_block (per_token + attention is incoherent).
        if is_v3 and self.dram.enabled and self.dram.routing_granularity != "per_block":
            raise ValueError(
                f"variant={self.variant} mandates dram.routing_granularity=per_block; "
                f"per_token routing has incoherent attention semantics under recurrence."
            )

        # (4c) Middle-cell shape sanity.
        if self.dram.cell_type == "recurrent_middle_block":
            if self.dram.recurrent_cell_layers <= 0:
                raise ValueError(
                    "recurrent_middle_block cell_type requires "
                    "dram.recurrent_cell_layers > 0."
                )
            if self.dram.prefix_layers < 0 or self.dram.suffix_layers < 0:
                raise ValueError("prefix_layers / suffix_layers must be ≥ 0.")

        # (5) Kalman authority without kill pass — `can_control_router=true`
        # also requires the AIR to be enabled.
        if self.kalman.can_control_router and not self.dram.router.enabled:
            raise ValueError(
                "kalman.can_control_router=true requires dram.router.enabled=true."
            )

        # (6) Tropical disallowed in pure `recommended`.
        if self.variant == "recommended" and self.attention.tropical.enabled:
            raise ValueError(
                "Tropical attention is not allowed in variant=recommended; "
                "use recommended_plus_tropical_probe or original_bets_mini."
            )

        # (7) RG-flow loss disallowed in pure `recommended`.
        if self.variant == "recommended" and self.dram.rg_flow.enabled:
            raise ValueError(
                "RG-flow loss is not allowed in variant=recommended (diagnostic-only). "
                "Use original_bets_mini if you want it."
            )

        # (8) FLOP convention.
        if self.compute.flop_convention != "6N_per_token":
            raise ValueError(
                "v3.1 mandates compute.flop_convention=6N_per_token; "
                f"got {self.compute.flop_convention!r}."
            )

        # (9) AIR training mode must be a known mode.
        if self.dram.router.enabled and self.dram.router.training_mode not in (
            "straight_through_soft_k",
            "controller_only",
        ):
            raise ValueError(
                "dram.router.training_mode must be 'straight_through_soft_k' or "
                f"'controller_only'; got {self.dram.router.training_mode!r}."
            )

        # (10) passive_probe Kalman must have a fusion_gate_init.
        if self.kalman.enabled and self.kalman.mode == "passive_probe":
            if self.kalman.fusion_gate_init is None:
                raise ValueError(
                    "kalman.mode=passive_probe requires fusion_gate_init (numeric)."
                )

        # (11) KL direction named explicitly.
        if self.training.distill_enabled:
            if self.training.distill_kl_direction not in (
                "teacher_forward_kl", "reverse_kl",
            ):
                raise ValueError(
                    "training.distill_kl_direction must be 'teacher_forward_kl' or "
                    f"'reverse_kl'; got {self.training.distill_kl_direction!r}."
                )

        # (12) Top-K logit storage requires logsumexp for normalized KL.
        if self.training.distill_enabled and self.training.distill_mode == "teacher_token_kl":
            if not self.training.distill_store_indices or not self.training.distill_store_logits:
                raise ValueError(
                    "teacher_token_kl distillation requires "
                    "distill_store_indices=true AND distill_store_logits=true."
                )
            if not self.training.distill_store_teacher_logsumexp:
                raise ValueError(
                    "teacher_token_kl with truncated top-K requires "
                    "distill_store_teacher_logsumexp=true for normalized KL."
                )

        # (12b) v3.1 recommended variant requires BPE tokenizer (alignment with teacher).
        if self.variant == "recommended" and self.tokenizer.type != "bpe":
            raise ValueError(
                "variant=recommended mandates tokenizer.type=bpe (teacher-aligned vocab); "
                f"got {self.tokenizer.type!r}. Use original_bets_mini for the BLT path."
            )

        # (12c) BPE tokenizer.type ⇒ vocab_size must match tokenizer's, and patcher off.
        if self.tokenizer.type == "bpe":
            from sprl.tokenizer import vocab_size_for

            try:
                want = vocab_size_for(self.tokenizer.source)
            except ValueError:
                want = None
            if want is not None and self.vocab_size != want:
                raise ValueError(
                    f"tokenizer.type=bpe with source={self.tokenizer.source!r} requires "
                    f"vocab_size={want}; got vocab_size={self.vocab_size}."
                )
            if self.patcher.enabled:
                raise ValueError(
                    "tokenizer.type=bpe is incompatible with patcher.enabled=true. "
                    "Disable BLT (patcher.enabled=false) when using BPE."
                )

        # (13) Tokenizer alignment for direct teacher-token KL.
        # The patcher (BLT) yields a 256-byte vocab; teacher BPE is 32K-152K.
        # Already covered above (BLT + teacher_token_kl). For non-BLT student
        # vocabs, alignment must be explicit via `vocab_size`.
        if (
            self.training.distill_enabled
            and self.training.distill_mode == "teacher_token_kl"
            and not self.patcher.enabled
            and self.vocab_size <= 1024
        ):
            raise ValueError(
                "teacher_token_kl requires student/teacher vocab alignment. "
                f"Student vocab_size={self.vocab_size} is too small for any common "
                "teacher; use auxiliary_teacher_token_head instead."
            )

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
