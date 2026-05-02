# Variants

The repository ships two operating variants plus a fallback. They share
the same code; only YAML config flags differ. Pick a variant via the
`--variant` shortcut or by passing an explicit `--config`.

```bash
# Pilot kill-experiments (~4h on a 5080 each):
python scripts/run_pilot.py --variant recommended
python scripts/run_pilot.py --variant original_bets_mini

# Full Stage-3 runs:
python scripts/run_full_pretrain.py --variant recommended --steps_per_phase 100
python scripts/run_full_pretrain.py --variant original_bets_mini --steps_per_phase 100

# Override entirely:
python scripts/run_pilot.py --config configs/my_custom.yaml
```

`recommended` is the default if `--variant` and `--config` are both omitted.

## `recommended` (default)

Practical version optimized for stable training. Defaults:

| Mechanism | Default |
|---|---|
| Tokenizer | BLT byte-level entropy patcher |
| DSA-MLA + sliding-window alternating | on |
| Fine-grained MoE + ALF balancing | on |
| MTP head | on |
| Depth recurrence (DRAM) | on, target K=2 |
| Active-inference router | **off** — entropy router fallback |
| Kalman info-form memory | **passive probe**: state computed + log_det logged but does NOT control the router |
| Tropical attention heads | **off** |
| RG-flow regularizer | **off** (diagnostic via `T_b_norm` log only) |
| Distillation | **off** by default; opt-in. Under BLT, must use `sequence_level` mode (validation errors on `teacher_token_kl` + BLT). |
| Active selection (Bet C) | off (requires `distill_enabled=true`) |
| Iterative re-distillation | off |

**Promotion rule**: if Bet A's kill criterion (Spearman ρ(-log_det Λ, oracle surprise) ≥ 0.5) passes at the 100M pilot, flip `dram.router.enabled: true` to give Kalman uncertainty authority over per-token K\*.

Configs:
- `configs/recommended_100m_pilot.yaml` — ~100M-active for ~4-hour kill experiments
- `configs/recommended_500m_full.yaml` — ~500M-active / ~50B tokens / ≤ 1 week 5080 BF16

## `original_bets_mini`

Research-ablation variant: keeps the original SPRL-v2 mechanisms (KalmanMemory active, tropical heads, RG-flow as diagnostic) at mini scale. **Not expected to be the strongest benchmark model** — the point is to falsify mechanisms.

Differences from the v2 spec:
- Tropical heads: **1 of 8** (~12.5%), not 1 of 4 — too expensive at mini.
- RG-flow: diagnostic-only (loss disabled). Flip `dram.rg_flow.enabled: true` only after the depth-dynamics power-law diagnostic stays stable across checkpoints.
- BLT + direct teacher-token-KL is hard-blocked by validation. If you want distillation, switch `distill_mode` to `sequence_level`.

Configs:
- `configs/original_bets_mini_100m_pilot.yaml` — kill experiments
- `configs/original_bets_mini_300m_full.yaml` — ~300M-active full run

## `fallback_boring`

All three novel Bets off. Boring SOTA backbone only (DSA-MLA + sliding window + MoE + MTP + BLT + NCA pre-pre-training). Use as the safety net if every Bet kill criterion fails — the boring backbone is still a credible ~SmolLM2-class model.

Config: `configs/fallback_boring.yaml`.

## Validation

`SPRLConfig.from_yaml()` runs `cfg.validate()` after parsing. Hard errors:

1. **BLT + `teacher_token_kl`**: incoherent (256-byte vocab vs 32K teacher BPE). Set `distill_mode` to `sequence_level` or disable BLT.
2. **Recurrence on but `count_recurrence_in_active_flops: false`**: throughput / FLOPs reports would silently undercount by ≈ mean_k×.
3. **`active_selection_enabled: true` without `distill_enabled: true`**: selection scores docs by student↔teacher KL and needs a teacher.
4. **`iterative_redistill_enabled: true` without `distill_enabled: true`**: same reason.

## Compute accounting

`active_params_per_token` is the standard convention (per-token, no recurrence multiplier) — keeps comparisons to published baselines honest. `effective_flops_per_token = 6 × active_params × mean_recurrent_iterations` is the *additional* number we report (when `count_recurrence_in_active_flops: true`).

```python
from sprl.bench import estimate_compute, estimate_memory_breakdown
from sprl.config import SPRLConfig
from sprl.model import SPRLv2

cfg = SPRLConfig.from_yaml("configs/recommended_500m_full.yaml")
m = SPRLv2(cfg)
print(estimate_compute(m, mean_recurrent_iterations=2))
print(estimate_memory_breakdown(m, batch=16, seq_len_patches=1024))
```

## Memory accounting

`estimate_memory_breakdown` reports static budgets for: weights, gradients, optimizer states, activations (with gradient checkpointing every 2 layers), KV cache, Kalman state, MoE dispatch buffers, teacher-logit buffers (when distillation is on), workspace. CLI:

```bash
python -m sprl.bench.memory_scaling --variant recommended
```
