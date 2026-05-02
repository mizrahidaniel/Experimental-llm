# Experiment log

This log is the durable record of kill-criterion outcomes, distillation
cutoff used, NCA pre-pre-training effect, and ablation results. It is meant
to be appended to as the project runs.

The entries below describe the **scaffolding state** as of the first commit,
not full-scale training results. A real run on a 5080 fills in the empty
columns.

## Stage 0 — μP HP search

| Run | LR | Init scale | Router-λ | RG-α | Tropical β_max | Notes |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

## Stage 1 — Pilot kill experiments (200M, 50B FineWeb-Edu)

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
