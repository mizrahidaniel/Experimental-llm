# Experiment log

This log is the durable record of kill-criterion outcomes, distillation
cutoff used, NCA pre-pre-training effect, and ablation results. It is meant
to be appended to as the project runs.

The entries below describe the **scaffolding state** after the parallel-agent
build-out, not full-scale training results. A real run on a 5080 fills in
the empty columns.

## Build-out state

After parallel-agent expansion the repository contains:

| Component | Status |
|---|---|
| Boring backbone (MLA + DSA + HISA + sliding + DRAM + MoE + MTP + byte-LM) | implemented + tested |
| Bet A (Kalman info-form memory + active-inference router) | implemented; strict-rank streaming-SVD path on by default; TTT-mode for inference |
| Bet B (tropical heads with softmax-large-β warmup) | implemented; β anneal scheduled |
| Bet C (RG-flow regularizer with rank-32 Marchenko-Pastur init) | implemented; ‖T_b−I‖_F + power-law diagnostics wired |
| BLT entropy patcher | implemented; byte-LM oracle is randomly-init (must be pretrained on 30B bytes before real use) |
| NCA pre-pre-training | implemented; LSH and VQ-codebook tokenizers both available |
| Distillation | top-32 sparse-logit NPZ format + resumable precompute script + sparse-KL loss + scheduler with hard cutoff |
| Curriculum | perplexity-correlation reweighting; frozen during MoE warmup |
| Precision | NVFP4 emulation + RHT + GaLore + 8-bit-AdamW shim; BF16 default |
| Token-level depth recurrence | per-token K* drives `TokenLevelDRAMBlock` masked iteration |
| Inference | speculative decoding with MTP head, KV cache, sampler, chat templates |
| Data pipeline | 7 source loaders + `DomainMixer` with live curriculum weights; long-context 64K loader; offline synthetic fallbacks |
| Eval | MMLU/GSM8K/HumanEval/GPQA/RULER/S5 + lm-eval-harness adapter |
| Bench | memory scaling / throughput / decode speed |
| Tests | **223 passing** on CPU in ~25s |

## Stage 0 — μP HP search

| Run | LR | Init scale | Router-λ | RG-α | Tropical β_max | Notes |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

## Stage 1 — Pilot kill experiments (100M, ~1B FineWeb-Edu, ~4 hours each)

The realistic plan runs the kill experiments at **100M active params on
~1B tokens**, taking ~4 hours of 5080 BF16 wall-clock per experiment, so
all three can complete within 24 hours. The 200M / 50B-token configs from
the original v2 spec (`configs/pilot_200m_*.yaml`) remain shipped for
anyone who has the budget.

### Kill experiment A (Bet A — Kalman info-form memory)

| Metric | Threshold | Measured | Pass |
|---|---|---|---|
| Spearman ρ(log\|Λ\|, oracle surprise) | ≥ 0.5 | _pending_ | _pending_ |
| Val loss vs Mamba-3 baseline | within ±0.2% | _pending_ | _pending_ |

Decision: _pending_.

### Kill experiment B (Bet B — Tropical lane)

| Metric | Threshold | Measured | Pass |
|---|---|---|---|
| GSM8K @ 0.5B tokens vs softmax-only | +3% | _pending_ | _pending_ |
| MMLU @ 0.5B tokens vs softmax-only | within ±0.5% | _pending_ | _pending_ |
| Argmax position concentration | < 0.95 | _pending_ | _pending_ |

Decision: _pending_.

### Kill experiment C (Bet C — RG-flow)

| Metric | Threshold | Measured | Pass |
|---|---|---|---|
| Power-law fit R² (loss vs iter) | ≥ 0.9 | _pending_ | _pending_ |
| ‖T_b − I‖_F | ≥ 0.1 | _pending_ | _pending_ |
| State-tracking S5 over no-RG | +10% | _pending_ | _pending_ |

Decision: _pending_.

## Stage 2 — Decision point (week 3)

The bet-survival graph (per spec §6.3):

| Survivors | Resulting model |
|---|---|
| A, B, C all pass | Full SPRL-v2 |
| A only | KalmanMemory + plain MLA + plain DRAM |
| B only | Plain GLA / DeltaNet + tropical heads |
| C only | Plain GLA / DeltaNet + RG-scheduled DRAM |
| none | Mamba-3 + DSA + MoE + MTP "boring SOTA" |

Decision: _pending_.

## Stage 3 — Full pretraining (≤ 6 weeks of 5080 wall-clock)

### Phase 3a — Distillation

| Metric | Value |
|---|---|
| Teacher | _pending_ |
| Tokens distilled | _pending_ |
| Distillation cutoff actually used | _pending_ |
| KL weight schedule | _pending_ |

### Phase 3b — Synthetic + verifier-RL

| Metric | Value |
|---|---|
| Phi-4 synthetic data fraction | _pending_ |
| rStar-Math MCTS traces tokens | _pending_ |

### Phase 3c — Long context

| Metric | Threshold | Measured |
|---|---|---|
| RULER@32K | ≥ 80 | _pending_ |
| RULER@128K | ≥ 60 | _pending_ |

### Phase 3d — Post-training

| Stage | Method | Tokens | Notes |
|---|---|---|---|
| SFT | Tulu-3 / Magpie-Pro | _pending_ | |
| DPO | _pending_ | _pending_ | |
| GRPO | _pending_ | _pending_ | |

## Final benchmark suite

| Benchmark | Pass threshold | Stretch | Measured | Pass |
|---|---|---|---|---|
| MMLU 5-shot | 65 | 70 | _pending_ | _pending_ |
| HellaSwag | 80 | 84 | _pending_ | _pending_ |
| ARC-Challenge | 60 | 65 | _pending_ | _pending_ |
| GSM8K 8-shot | 80 | 88 | _pending_ | _pending_ |
| MATH | 35 | 50 | _pending_ | _pending_ |
| HumanEval | 65 | 78 | _pending_ | _pending_ |
| GPQA-diamond | 30 | 38 | _pending_ | _pending_ |
| RULER@32K | 80 | 90 | _pending_ | _pending_ |
| RULER@128K | 60 | 80 | _pending_ | _pending_ |

## Ablation matrix

| Ablation | Result |
|---|---|
| Bet A on/off | _pending_ |
| Bet B on/off | _pending_ |
| Bet C on/off | _pending_ |
| DSA on/off | _pending_ |
| MTP on/off | _pending_ |
| NCA pre-pre-train on/off | _pending_ |
| Distillation cutoff {0%, 30%, 60%, 100%} | _pending_ |
| MoE granularity {16, 32, 64, 128} experts | _pending_ |
| Heterogeneous vs homogeneous experts | _pending_ |
| NVFP4 vs FP8 vs BF16 | _pending_ |
