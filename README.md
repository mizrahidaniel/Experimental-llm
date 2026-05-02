# SPRL-v2

**EXPERIMENTAL ARCHITECTURE PROTOTYPE.**

This repository implements **SPRL-v2** (Sparse-Patch Recurrent Latent), an
experimental decoder-only LM designed to train and run on a single RTX 5080
(16 GB).

## Scale targets

The architecture supports two operating points; the codebase ships configs
for both. The realistic single-5080 plan is the smaller one.

| Scale | Active params | Total params | Tokens | Wall-clock | Config |
|---|---|---|---|---|---|
| **Realistic** (default plan) | ~500M | ~2B | 50B | ≤ 1 week BF16 | `configs/full_500m_bf16.yaml` |
| Original ambition (v2 spec) | 1.5–2B | 16–24B | 1T | ~6 weeks | not shipped — see `docs/architecture.md` |
| Stage-1 pilot (kill experiments) | ~80M | ~245M | 1B | ~4 hours per experiment | `configs/pilot_100m_*.yaml` |

**Local training only**: the default plan does not use teacher
distillation. The distillation pipeline (`sprl/training/teacher_logits.py`,
`scripts/precompute_teacher_logits.py`) is shipped but disabled by default;
flip `training.distill_enabled: true` to use precomputed top-32 NPZ teacher
logits.

It is **NOT proven to outperform Transformers.** It combines:

- DeepSeek V3.2-style DSA-under-MLA sparse attention
- Depth-recurrent DRAM core (weight-tied iterative reasoning)
- Fine-grained DeepSeek-V3 MoE with auxiliary-loss-free balancing
- Multi-token prediction head
- NVFP4 / Quartet native FP4 training (emulated by default; real FP4 path
  requires `transformer-engine ≥ 2.0` on Blackwell)
- BLT entropy patching
- NCA pre-pre-training (free 3% loss reduction)

Plus three concentrated **novel bets**:

- **Bet A**: Kalman information-form belief memory (replaces PIG router AND
  Titans-style test-time MLP).
- **Bet B**: Dual-algebra (softmax + tropical) attention heads (1 in 4 heads
  runs in max-plus algebra).
- **Bet C**: RG-scheduled depth recurrence regularizer.

Each novel bet has a **kill criterion** that must pass at 200M parameters
before scaling to 1.5–2B active.

This repository is designed to test these ideas, not to claim success without
evidence. See `docs/red_team.md` for known failure modes,
`docs/novelty_audit.md` for prior-art comparison, and
`docs/experiment_log.md` for kill-criterion outcomes.

For deployment on knowledge-heavy benchmarks (GPQA, MMLU long-tail), pair
with a retrieval index — Allen-Zhu's 2-bit/parameter ceiling is binding for
a small parametric model.

---

## What's in this repository

This is the **architectural scaffold + unit-tested PyTorch reference**. It
includes:

- A **complete, runnable** SPRL-v2 model in PyTorch that trains end-to-end
  on synthetic data on a CPU or any GPU.
- **All three Bets** (A, B, C) implemented as opt-in modules.
- **223 unit tests** verifying the architectural invariants from spec
  Section 8.1 — see `tests/`. All tests run on CPU in ~25 seconds.
- **Kill-criterion harness** (`scripts/run_kill_experiments.py`) and
  diagnostic primitives (Spearman ρ, power-law fit, ‖T_b − I‖, argmax
  concentration) in `sprl/utils/diagnostics.py`.
- **Configs** for the pilot 200M run and each kill experiment in `configs/`.
- **Precision policy** (NVFP4 emulated forward, GaLore, RHT, BF16 fallback)
  in `sprl/precision/`.
- **Training scaffolding** (loss assembly, schedules, distillation,
  perplexity-correlation curriculum) in `sprl/training/`.
- **Real data pipeline** (`sprl/data/`): FineWeb-Edu / DCLM / StarCoder /
  math-pile / Phi-4-style synthetic / rStar-Math verified MCTS / NCA
  trajectories / 64K long-context loaders, plus a deterministic
  `DomainMixer` that consumes live `PerplexityCorrelationCurriculum` weights.
  HuggingFace `datasets` is optional; offline synthetic fallbacks ship.
- **Offline teacher distillation** (`sprl/training/teacher_logits.py`):
  top-32 sparse-logit NPZ shard format, resumable precomputation script
  (`scripts/precompute_teacher_logits.py`), `TeacherLogitDataset` streaming
  IterableDataset with prefetch + seek, exact-at-K=V sparse-KL loss.
- **NCA VQ codebook** (`sprl/training/nca_pretrain.py::VQCodebook`):
  L2-normalized EMA-updated centroids per arXiv 2106.13409, replacing the
  legacy LSH tokenizer (kept as fallback).
- **Inference subpackage** (`sprl/inference/`): MLA KV-cache, byte-level
  sampler with temperature/top-p/top-k, MTP speculative decoding with
  acceptance-rate accounting, chat-template scaffolding, and a `generate()`
  driver. TTT-mode KalmanMemory (`sprl/memory/ttt_mode.py`) lets the Bet-A
  filter adapt online during inference.
- **Eval harness** (`sprl/eval/`): MMLU 5-shot, GSM8K 8-shot CoT, HumanEval
  pass@1 (subprocess sandbox), GPQA-diamond, RULER@8K/32K/128K (synthetic
  needles in-process), S5 state-tracking (Bet C kill diagnostic),
  byte-level perplexity / bits-per-byte, plus `SPRLHarnessAdapter` matching
  the lm-eval-harness 0.4.x API.
- **Benchmarks** (`sprl/bench/`): memory scaling, fwd+bwd throughput,
  decode tokens/sec with and without MTP speculation.
- **Numerical strict-rank Kalman** (`sprl/memory/kalman_info.py` with
  `strict_rank=True`, `sprl/memory/streaming_svd.py`): info-form recurrence
  with growing-rank concatenation and Brand-2002 streaming-SVD truncation
  at chunk boundaries — associative under (concat, then-truncate). Legacy
  rank-truncated combine kept for ablations.
- **Work-efficient parallel scan** (`sprl/memory/parallel_scan.py::blelloch_scan`):
  bit-identical to the sequential reference; identity-free via a `valid`
  mask; per-level batched combines.
- **Token-level depth recurrence** (`sprl/recurrent/dram_block.py::TokenLevelDRAMBlock`):
  the active-inference router's per-token K* now actually controls
  per-token compute via masked iteration freezing.
- **CI** (`.github/workflows/tests.yml`): Python 3.10/3.11 matrix, ruff
  lint (report-only), pytest.

What this repository does **NOT** include (you must supply):

- A real Blackwell training run. Doing so requires:
  - Real BLT-tokenized FineWeb-Edu / DCLM data shards.
  - `transformer-engine`, `flash-attn-3`, `flash-linear-attention`,
    `bitsandbytes`, `mamba-ssm`, `triton` installed against a CUDA 12.8 build.
  - A teacher (Llama-3.3-70B / DeepSeek-V3) for offline logit precomputation
    (the orchestration script and storage format are shipped — point
    `scripts/precompute_teacher_logits.py` at a real model).
  - Multi-day to multi-week wall-clock on a 5080.
- The actual kill-criterion outcomes at 50B tokens. The harness, the
  diagnostics, the eval suite, and the data pipeline are wired up; the
  experiments must be run.

---

## Reproduction quick-start

```bash
pip install -e .[test]

# Unit tests (CPU-only, fast)
pytest tests/

# Pilot smoke test — instantiate the 200M config, drive 2 steps of synthetic
# data through it, verify the loss decreases. Replace with your real loader.
python scripts/run_pilot.py --config configs/pilot_200m_bf16.yaml --steps 2

# Kill-experiment harness (synthetic; replace with FineWeb-Edu loader for the
# real measurement)
python scripts/run_kill_experiments.py --bet all --steps 4
```

---

## Repository structure

```
sprl/
  config.py                # dataclass-based hierarchical config
  model.py                 # SPRLv2 main module (token-level DRAM)
  blocks.py                # SPRLLayer + attention/FFN factories
  patcher/                 # BLT entropy patcher + byte encoder/decoder
  attention/               # MLA, DSA, HISA, sliding window, tropical (Bet B)
  recurrent/               # DRAM block + TokenLevelDRAMBlock,
                           #   depth-attention, RG flow (Bet C),
                           #   active-inference router
  memory/                  # Kalman info-form memory (Bet A) with strict-rank
                           #   streaming-SVD path, parallel scan + Blelloch,
                           #   low-rank precision algebra, ttt_mode
  experts/                 # fine-grained MoE, ALF balancer, top-k router
  heads/                   # MTP head, byte LM head
  precision/               # NVFP4 emulation, RHT, GaLore, dtype policy
  training/                # loss assembly, distillation (sparse top-K KL),
                           #   teacher_logits (NPZ shard format), curriculum,
                           #   NCA + VQ codebook, schedules, optimizer, train
  data/                    # FineWeb-Edu / DCLM / StarCoder / math-pile /
                           #   synthetic_phi4 / rstar_math / nca_trajectories /
                           #   long_context loaders, mixer, loader factory
  inference/               # MLA KV-cache, sampler, MTP speculative decoding,
                           #   chat templates, generate driver
  eval/                    # MMLU / GSM8K / HumanEval / GPQA / RULER /
                           #   state_tracking (S5) / perplexity / harness adapter
  bench/                   # memory_scaling, throughput, decode_speed
  utils/                   # diagnostics, logging, seeds

configs/
  pilot_100m_bf16.yaml         # 100M baseline for fast kill experiments
  pilot_100m_kill_exp_A.yaml   # Bet A on (Kalman info-form memory)
  pilot_100m_kill_exp_B.yaml   # Bet B on (tropical heads)
  pilot_100m_kill_exp_C.yaml   # Bet C on (RG-flow regularizer)
  pilot_200m_bf16.yaml         # 200M baseline (legacy v2 pilot)
  pilot_200m_kill_exp_A.yaml   # Bet A on at 200M
  pilot_200m_kill_exp_B.yaml   # Bet B on at 200M
  pilot_200m_kill_exp_C.yaml   # Bet C on at 200M
  full_500m_bf16.yaml          # 500M-active full Stage-3 target (realistic plan)
  fallback_boring.yaml         # all bets off (safety-net config)

data/fixtures/             # Tiny offline JSONL fixtures for synthetic_phi4
                           #   and rstar_math so the data tests run with no
                           #   network.

tests/                     # 223 unit tests, all on CPU (~25s)
scripts/                   # run_pilot.py, run_kill_experiments.py,
                           # run_full_pretrain.py, run_data_smoke.py,
                           # run_eval_suite.py, precompute_teacher_logits.py
docs/                      # architecture.md, novelty_audit.md, red_team.md,
                           # experiment_log.md
.github/workflows/tests.yml  # CI: ruff (report-only) + pytest matrix
```

---

## Kill criteria recap (from spec §6.2)

| Bet | Criterion |
|---|---|
| A | Spearman ρ(log\|Λ_t\|, oracle teacher surprise) ≥ 0.5 AND val loss within ±0.2% of Mamba-3 baseline at matched FLOPs |
| B | GSM8K +3% over softmax-only baseline AND MMLU within ±0.5% |
| C | Loss-vs-iterations power-law R² ≥ 0.9 AND ‖T_b − I‖_F ≥ 0.1 AND S5 / state-tracking +10% |

If any bet fails, demote it (see spec §6.3). Do **not** tune past a kill
criterion.

## License

Apache-2.0.
