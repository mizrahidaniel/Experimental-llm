# Variants

The repository ships three operating variants plus a fallback. They share
the same code; only YAML config flags differ. Pick a variant via the
`--variant` shortcut or by passing an explicit `--config`.

```bash
# Pilot kill-experiments (~4h on a 5080 each):
python scripts/run_pilot.py --variant recommended
python scripts/run_pilot.py --variant recommended_plus_tropical_probe
python scripts/run_pilot.py --variant original_bets_mini

# Full Stage-3 runs:
python scripts/run_full_pretrain.py --variant recommended --steps_per_phase 100
python scripts/run_full_pretrain.py --variant original_bets_mini --steps_per_phase 100

# Override entirely:
python scripts/run_pilot.py --config configs/my_custom.yaml
```

`recommended` is the default if `--variant` and `--config` are both omitted.

## `recommended` (default — v3.1)

Practical version optimized for stable training. Defaults:

| Mechanism | Default |
|---|---|
| Tokenizer | BLT byte-level entropy patcher |
| DSA-MLA + sliding-window alternating | on |
| Fine-grained MoE + ALF balancing | on |
| MTP head | on |
| **Cell layout** | `recurrent_middle_block`: prefix(2) + tied-cell(4)·K + suffix(2) |
| **Routing granularity** | `per_block` — one K per batch element |
| Active-inference router | on, `straight_through_soft_k` mode, `scout_head_only` source |
| Kalman info-form memory | **passive_probe** mode with gated residual fusion `h += sigmoid(fusion_gate)·kalman_out`, `fusion_gate_init = -3.0` (sigmoid ≈ 0.05) — does NOT control the router |
| Tropical attention heads | **off** |
| RG-flow regularizer | **off** (diagnostic via `T_b_norm` log only) |
| Distillation | **off** by default; opt-in. Under BLT, must use `auxiliary_teacher_token_head` (256-byte vocab can't be aligned with a 128K teacher BPE). |
| Active selection (Bet C) | off (requires `distill_enabled=true`) |
| Iterative re-distillation | off |
| FLOP convention | `6N_per_token` × `architecture_overhead_factor` (default 1.3) |

**Promotion rule**: if Bet A's kill criterion (Spearman ρ(-log_det Λ, oracle surprise) ≥ 0.5) passes at the 100M pilot, flip `kalman.can_control_router: true` AND `dram.router.source: kalman_plus_scout_if_authority_else_scout_only` to give Kalman uncertainty authority over per-block K\*.

Configs:
- `configs/recommended_100m_pilot.yaml` — ~100M-active for ~4-hour kill experiments
- `configs/recommended_300m_full.yaml` — ~250-350M *effective active* / target ~30B tokens / ≤ 1 week 5080 BF16 (FP8 stretch ~50B; NVFP4 aspirational ~100B)

## `recommended_plus_tropical_probe`

Identical to `recommended` except one tropical head is added (≈ 1/12 ≈ 8% of heads, capped at 1/8 by the `never_exceed_fraction` ceiling). Use to test whether tropical attention helps without polluting the default path. Pass criterion: improves synthetic DP/state-tracking tasks AND does not regress LM val loss by > 1.0% vs `recommended`.

Config: `configs/recommended_plus_tropical_probe_pilot.yaml` (pilot only — no full-run config by design; promote tropical into a follow-up `recommended` iteration if it passes).

## `original_bets_mini` (v3.1 ablation testbed)

Research-ablation variant: keeps the original SPRL-v2 mechanisms (KalmanMemory active, tropical heads, RG-flow as diagnostic) at mini scale. **Not expected to be the strongest benchmark model** — the point is to falsify mechanisms.

Differences from the v2 spec:
- Tropical heads: **1 of 12** (~8.3%), capped at 1/8 (`never_exceed_fraction = 0.125`).
- RG-flow: diagnostic-only (loss disabled). Flip `dram.rg_flow.enabled: true` only after the depth-dynamics power-law diagnostic stays stable across checkpoints.
- BLT + direct teacher-token-KL is hard-blocked by validation. Default `distill_mode: auxiliary_teacher_token_head` so a teacher with mismatched vocab works.
- Kalman runs in `active_if_kill_passes` mode with a higher initial gate (`fusion_gate_init = -2.0`, sigmoid ≈ 0.12).

Configs:
- `configs/original_bets_mini_100m_pilot.yaml` — kill experiments
- `configs/original_bets_mini_300m_full.yaml` — full ablation run

## `fallback_boring`

All three novel Bets off. Boring SOTA backbone only (DSA-MLA + sliding window + MoE + MTP + BLT + NCA pre-pre-training). Use as the safety net if every Bet kill criterion fails — the boring backbone is still a credible ~SmolLM2-class model.

Config: `configs/fallback_boring.yaml`.

## Validation

`SPRLConfig.from_yaml()` runs `cfg.validate()` after parsing. The full v3.1 hard-check matrix:

1. **BLT + `teacher_token_kl`**: incoherent (256-byte vocab vs ≥32K teacher BPE). Set `distill_mode` to `auxiliary_teacher_token_head` or `sequence_level`, or disable BLT.
2. **Recurrence on but `count_recurrence_in_active_flops: false`**: throughput / FLOPs reports would silently undercount by ≈ mean_k×.
3. **`active_selection_enabled: true` without `distill_enabled: true`**: selection scores docs by student↔teacher KL and needs a teacher.
4. **`iterative_redistill_enabled: true` without `distill_enabled: true`**: same reason.
5. **v3.1 variant + `cell_type ≠ recurrent_middle_block`**: v3.1 mandates the prefix+cell+suffix layout.
6. **v3.1 variant + `routing_granularity ≠ per_block`**: per-token routing is incoherent under attention.
7. **`recurrent_middle_block` cell with `recurrent_cell_layers ≤ 0`**: degenerate shape.
8. **`kalman.can_control_router=true` without `dram.router.enabled=true`**: nothing to control.
9. **`recommended` variant + tropical / RG-loss enabled**: those belong in the probe / original-bets variants.
10. **`compute.flop_convention ≠ 6N_per_token`**: v3.1 standardizes on Chinchilla 6N.
11. **AIR `training_mode` ∉ {`straight_through_soft_k`, `controller_only`}**: unknown training mode.
12. **`kalman.mode = passive_probe` without `fusion_gate_init`**: passive probe MUST have a gated output path so the gate gets gradients.
13. **`distill_kl_direction` ∉ {`teacher_forward_kl`, `reverse_kl`}**: name the KL direction explicitly.
14. **`teacher_token_kl` distillation without `distill_store_teacher_logsumexp=true`**: top-K KL needs the logsumexp for normalization.
15. **`teacher_token_kl` with student vocab ≤ 1024 (non-BLT)**: student vocab too small to align with any common teacher.

## Compute accounting

`active_params_per_token` is the standard convention (per-token, no recurrence multiplier) — keeps comparisons to published baselines honest. The v3.1 `effective_active_params_per_token` and `effective_flops_per_token` are the *additional* numbers we report:

```
effective_active = prefix_active + cell_active × mean_k + suffix_active   (middle-cell)
                 = active_params × mean_k                                  (uniform layers)
effective_flops_per_token = 6 × effective_active × architecture_overhead_factor
```

```python
from sprl.bench import estimate_compute, estimate_memory_breakdown
from sprl.config import SPRLConfig
from sprl.model import SPRLv2

cfg = SPRLConfig.from_yaml("configs/recommended_300m_full.yaml")
m = SPRLv2(cfg)
print(estimate_compute(m, mean_recurrent_iterations=2))
print(estimate_memory_breakdown(m, batch=16, seq_len_patches=1024))
```

See `docs/compute_accounting.md` for the full convention.

## Distillation alignment

`teacher_token_kl` requires identical student/teacher tokenizers. With BLT (256-byte vocab) you must use `auxiliary_teacher_token_head` instead — see `docs/distillation_alignment.md`.
