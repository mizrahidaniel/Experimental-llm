# Compute accounting (v3.1)

Honest compute / memory accounting is mandatory under v3.1. The estimator
in `sprl/bench/memory_scaling.py` enforces the conventions below; validation
in `SPRLConfig.validate()` errors out when accounting silently undercounts.

## FLOP convention

```
train_flops_per_token = 6 × effective_active_params × architecture_overhead_factor
```

The `6N` factor is the standard Chinchilla scaling-law convention for full
forward + backward training cost (2N forward + 4N backward).

The `architecture_overhead_factor` accounts for everything the bare 6N misses:
MoE routing, sparse-attention indexer cost, recurrence boilerplate, MTP head,
data loading, eval, teacher logit I/O. **Default 1.3**, realistic range 1.3–1.8.
Update from measured throughput after the first 1B tokens.

**Hard rules** (enforced by `cfg.validate()`):
- `compute.flop_convention` MUST be `"6N_per_token"`. Other conventions (e.g.
  `forward = 6N`, `backward = 12N`, `total = 18N`) conflict and are rejected.
- If `dram.enabled` and `dram.k_iterations_default > 1`, then
  `compute.count_recurrence_in_active_params` MUST be `true`. Otherwise the
  estimator silently undercounts by ≈ mean_k×.

## Effective active parameters

For middle-cell layouts:

```
effective_active = prefix_active + cell_active × mean_recurrent_iterations + suffix_active
```

For uniform layouts (legacy):

```
effective_active = active_params × mean_recurrent_iterations  (when count_recurrence_in_active_flops=true)
                 = active_params                              (otherwise)
```

`active_params` is computed by subtracting the inactive MoE expert params from
the total — i.e. only the routed experts that fire for any given token count.

## What the estimator returns

```python
from sprl.bench import estimate_compute
out = estimate_compute(model, mean_recurrent_iterations=2)
```

Keys:
- `n_params_total` — every parameter, including inactive MoE experts.
- `active_params_per_token` — standard convention. Keep this for baseline comparison.
- `active_params_per_recurrent_step` — alias of `active_params_per_token`.
- `mean_recurrent_iterations`
- `flops_per_token_one_step` — `6 × active_params` (no overhead, no recurrence).
- `effective_active_params_per_token` — recurrence-aware count above.
- `effective_flops_per_token` — `6 × effective_active × architecture_overhead_factor`.
- `architecture_overhead_factor` — what the estimator was using.
- `prefix_active_params`, `cell_active_params`, `suffix_active_params` — the
  three buckets when middle-cell layout is in use.

## Memory accounting

`estimate_memory_breakdown(model, batch, seq_len_patches)` returns:

| Key | Description |
|---|---|
| `weights_gb` | Main model weights, BF16 |
| `gradients_gb` | Same shape as weights |
| `optimizer_states_gb` | 8-bit AdamW + GaLore (~2 bytes/param) |
| `activations_gb` | Gradient-checkpointed every 2 layers |
| `kv_or_attention_cache_gb` | MLA latent KV cache |
| `kalman_state_gb` | Per-token (η, D, U) when Bet A is on |
| `moe_buffers_gb` | Top-k routing dispatch scratch |
| `teacher_logits_gb` | Top-32 + indices buffer when distill is on |
| `workspace_gb` | PyTorch + NCCL + fragmentation slack |
| `estimated_total_gb` | Sum |

CLI:

```bash
python -m sprl.bench.memory_scaling --variant recommended
```

## Honest token budgets (corrected from v3)

At 110 TFLOPS sustained BF16, 6N × 1.3 overhead, 250M effective active:

| Tier | Tokens | Time |
|---|---|---|
| **BF16 prototype (safe)** | 30B | ~7 days ✓ |
| FP8 stretch (kernel-dependent) | 50B | ~7 days, only if FP8 measured-good |
| NVFP4 aspirational | 100B | ~7 days, only if FP4 stable |

The earlier 50B-tokens-in-1-week-BF16 claim was wrong by ~2×. Plan around
30B for BF16; only promote tiers after measuring.

## Recurrence pilot grid

`scripts/run_recurrence_grid.py` sweeps `(target_mean_k, max_k) ∈ {2,3} × {4,6}`
on 1B tokens each. Choose the (K, max_K) point that fits the 7-day budget
under the measured `architecture_overhead_factor`.
