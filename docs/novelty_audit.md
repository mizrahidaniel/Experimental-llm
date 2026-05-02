# Novelty audit

A line-by-line accounting of what is genuinely new in SPRL-v2 vs. existing
2025-2026 prior art.

## Boring backbone (all derivative)

- **DSA-under-MLA** — DeepSeek V3.2 (arXiv 2512.02556).
- **HISA hierarchical indexer** — arXiv 2603.28458.
- **InfLLM-V2 dense↔sparse switchable** — arXiv 2509.24663.
- **ASA alternating** — arXiv 2511.00819.
- **DRAM depth-recurrent core** — arXiv 2601.21582.
- **Mixture-of-Recursions / MoR** — arXiv 2507.10524.
- **Tropical attention head, in isolation** — arXiv 2505.17190 + 2601.09775.
- **Kalman Linear Attention / KLA** — arXiv 2602.10743.
- **Fine-grained DeepSeekMoE + ALF balancing** — arXiv 2412.19437 +
  arXiv 2512.03915.
- **Multi-token prediction (MTP)** head — DeepSeek-V3 (2412.19437).
- **NCA pre-pre-training** — arXiv 2603.10055.
- **Quartet / NVFP4** native FP4 training — arXiv 2505.14669 + 2509.25149.
- **BLT entropy patching** — arXiv 2412.09871.
- **Distillation cutoff at ~30%** — arXiv 2509.01649.
- **Phi-4 synthetic data + verifier-RL** — arXiv 2412.08905.
- **Perplexity-correlation curriculum** — arXiv 2409.05816.

We do not claim novelty for any of the above. Where we deviate from a
reference implementation, it is for engineering convenience or to fit a
single-5080 memory budget.

## Three concentrated novel bets

### Bet A — Kalman info-form memory + active-inference router

- **What is novel**: using the *same* Kalman state (η, Λ) for both (1) a
  Bayesian recurrent memory replacing Titans/MIRAS-style test-time MLP and
  (2) the compute-routing signal via free-energy minimization. Every prior
  work uses one mechanism for memory and a *different* mechanism for routing.
- **What is borrowed**: the information-form parallel-scan recurrence (KLA),
  low-rank precision parameterization (standard in robotics SLAM), the
  active-inference free-energy formulation (Friston).
- **Falsifiability**: Spearman ρ(log\|Λ\|, oracle surprise) ≥ 0.5 + Mamba-3
  loss parity. See `docs/red_team.md` §10.2 for the failure mode and
  `tests/test_unit_kill_criteria.py` for the diagnostic harness.

### Bet B — Dual-algebra (softmax + tropical) heads

- **What is novel**: head-partitioned dual-algebra MLA, where 1 of 4 heads
  is genuinely max-plus rather than just a high-temperature softmax variant.
  The softmax-with-large-β warmup smoothly anneals between the two.
- **What is borrowed**: tropical attention itself (arXiv 2505.17190, 2601.09775).
- **Falsifiability**: GSM8K +3% over softmax-only AND MMLU within ±0.5%. If
  GSM8K passes but MMLU regresses, demote tropical to inference-time only.

### Bet C — RG-scheduled depth recurrence

- **What is novel**: explicit T_b learnable block-spin operator regularizing
  z_{k+1} ≈ T_b z_k, with Marchenko-Pastur initialization and power-law
  loss-vs-iter diagnostic. Replaces LFR's σ_max(J_T) ≤ ρ contractivity penalty.
- **What is borrowed**: the RG framework for deep nets (arXiv 2510.25553);
  the empirical observation that latent fixed points emerge naturally
  (arXiv 2604.11791).
- **Falsifiability**: power-law fit R² ≥ 0.9 AND ‖T_b − I‖_F ≥ 0.1 AND
  state-tracking +10% over plain DRAM.

## Summary

Three bets. Each kill criterion is automatable (the diagnostics in
`sprl/utils/diagnostics.py` are unit-tested in `tests/test_diagnostics.py`).
If the kill criteria fail at 200M, the config flag for that bet flips to
`enabled: false` and we ship the boring backbone.

The "demotion graph" is documented in spec §6.3.
