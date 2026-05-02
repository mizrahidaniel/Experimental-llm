# Red team — known failure modes

These are the failure modes most likely to occur in a real run. Each maps
1:1 to spec Section 10.

## 10.1 Mechanism stacking

With 3 novel bets + ≥ 8 boring SOTA components, hyperparameter interaction
is the dominant risk.

**Mitigation**:
- Each bet is **off by default** in `sprl/config.py`. Enable per kill experiment.
- Kill experiments at 200M run *before* scaling.
- The `default_pilot_200m()` factory enables only the boring backbone.

## 10.2 KalmanMemory collapse

Λ_t collapsing to a near-constant scale (no informative uncertainty) is
known to occur in low-rank Bayesian filters.

**Mitigation**:
- `eps_diag` parameter adds a small diagonal each step to keep D ≻ 0.
- The diagnostic `log_det_Lambda_mean` is logged every step and surfaced in
  `SPRLOutput.diagnostics`.
- Kill criterion: if `std_t(log_det_Lambda) < 0.1 nats` across batch, demote
  to GLA / Gated DeltaNet. Implemented in
  `scripts/run_kill_experiments.py::kill_a`.

## 10.3 Tropical head dead pathway

Tropical heads converging to redundant pathways (effectively a
high-temperature softmax).

**Mitigation**:
- The `argmax_concentration` diagnostic in `sprl/utils/diagnostics.py`
  measures the fraction of positions where softmax-argmax == tropical-argmax.
  Threshold > 0.95 ⇒ tropical head is degenerate.
- Beta annealing (`sprl/training/schedules.py::beta_anneal`) drives β from 1
  to 32 over the warmup window so heads diverge from softmax over time.

## 10.4 RG-flow collapse

T_b → identity, regularizer becomes inactive, scheme collapses to ordinary
depth-recurrence.

**Mitigation**:
- Marchenko-Pastur init places singular values near the bulk edge, far from
  identity. See `tests/test_rg_regularizer.py::test_distance_from_identity_at_init`.
- Diagnostic `T_b_norm` (= ‖T_b − I‖_F) logged every step. Kill if < 0.1.
- The RG loss is non-zero only when the residual is non-zero, so a learning
  signal exists for both T_b and the recurrence.

## 10.5 Distillation overshoot

Distilling beyond 30% of tokens hurts ICL (arXiv 2509.01649).

**Mitigation**:
- `DistillScheduler` in `sprl/training/distill.py` hard-stops at the
  configured `cutoff_tokens` (default 300B), regardless of weight schedule.
- After the cutoff, KL weight is forced to 0 — the model returns to pure
  next-byte CE.

## 10.6 NVFP4 instability

Tropical heads, KalmanMemory updates may be unstable in pure FP4.

**Mitigation**:
- `sprl/precision/dtype_select.py` keeps "sensitive" modules
  (rmsnorm scales, kalman state, tropical warmup, lm_head) in BF16 even when
  the global default is FP4.
- The default config (`pilot_200m_bf16.yaml`) trains everything in BF16.
  Switch to NVFP4 only after the BF16 path matches a baseline.
- `tests/test_nvfp4_roundtrip.py` checks that emulated FP4 round-trip error
  is ≤ 10% Frobenius on synthetic activations.

## 10.7 MoE expert collapse under curriculum reweighting

ALF balancing assumes stationarity (arXiv 2512.03915); curriculum reweighting
violates it.

**Mitigation**:
- Small auxiliary balance loss (weight 0.001) is always available via
  `FineGrainedMoE.aux_loss()`.
- Curriculum is **frozen** until `freeze_until_tokens` (default 50B).
  Implemented in `PerplexityCorrelationCurriculum.maybe_update`.
- The `max_load_ratio` diagnostic is exposed on `FineGrainedMoE` for
  per-step monitoring.

## 10.8 Indexer mis-trained at short context

DSA's lightning indexer is hard to train if pretraining context is short.

**Mitigation**:
- `DynamicSparseAttention.dense_until_seqlen` runs *dense* attention below
  the threshold so the indexer trains alongside dense from the start
  (InfLLM-V2 switchable behavior). Default threshold = 1024.
- `tests/test_dsa.py::test_dsa_short_sequence_runs_dense` verifies the
  switch.

## 10.9 Truncated BPTT for KalmanMemory

Across-chunk gradient detach limits effective context for gradient flow.

**Mitigation**:
- The chunked associative scan in `sprl/memory/parallel_scan.py` does NOT
  detach by default — gradient flows through chunks. For TBPTT, the caller
  can `.detach()` the carry between chunks.
- Long-context training data (Phase 3c in spec) provides explicit signal.

## 10.10 GPQA / long-tail trivia are unfixable parametrically

Allen-Zhu's 2-bit/parameter ceiling is binding for a 1.2B-active model.

**Mitigation**:
- Document this as a structural cost; pair with retrieval at deployment.
- Do **not** chase GPQA by enlarging the model.
