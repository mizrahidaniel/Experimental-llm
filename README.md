# SPRL-v2

**EXPERIMENTAL ARCHITECTURE PROTOTYPE.**

This repository implements **SPRL-v2** (Sparse-Patch Recurrent Latent), an
experimental decoder-only LM designed to train and run on a single RTX 5080
(16 GB).

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
- **Unit tests** verifying the architectural invariants from spec Section 8.1
  — see `tests/`. All tests run on CPU.
- **Kill-criterion harness** (`scripts/run_kill_experiments.py`) and
  diagnostic primitives (Spearman ρ, power-law fit, ‖T_b − I‖, argmax
  concentration) in `sprl/utils/diagnostics.py`.
- **Configs** for the pilot 200M run and each kill experiment in `configs/`.
- **Precision policy** (NVFP4 emulated forward, GaLore, RHT, BF16 fallback)
  in `sprl/precision/`.
- **Training scaffolding** (loss assembly, schedules, distillation, perplexity-
  correlation curriculum) in `sprl/training/`.

What this repository does **NOT** include (you must supply):

- A real Blackwell training run. Doing so requires:
  - Real BLT-tokenized FineWeb-Edu / DCLM data shards.
  - `transformer-engine`, `flash-attn-3`, `flash-linear-attention`,
    `bitsandbytes`, `mamba-ssm`, `triton` installed against a CUDA 12.8 build.
  - A teacher (Llama-3.3-70B / DeepSeek-V3) for offline logit precomputation.
  - Multi-day to multi-week wall-clock on a 5080.
- The actual kill-criterion outcomes at 50B tokens. The harness is wired up;
  the experiments must be run.

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
  model.py                 # SPRLv2 main module
  blocks.py                # SPRLLayer + attention/FFN factories
  patcher/                 # BLT entropy patcher + byte encoder/decoder
  attention/               # MLA, DSA, HISA, sliding window, tropical (Bet B)
  recurrent/               # DRAM block, depth-attention, RG flow (Bet C),
                           #   active-inference router
  memory/                  # Kalman info-form memory (Bet A), parallel scan,
                           #   low-rank precision algebra
  experts/                 # fine-grained MoE, ALF balancer, top-k router
  heads/                   # MTP head, byte LM head
  precision/               # NVFP4 emulation, RHT, GaLore, dtype policy
  training/                # loss assembly, distillation, curriculum, NCA,
                           #   schedules, optimizer
  utils/                   # diagnostics, logging, seeds

configs/
  pilot_200m_bf16.yaml         # baseline (no novel bets)
  pilot_200m_kill_exp_A.yaml   # Bet A on
  pilot_200m_kill_exp_B.yaml   # Bet B on
  pilot_200m_kill_exp_C.yaml   # Bet C on

tests/                     # 76 unit tests, all on CPU
scripts/                   # run_pilot.py, run_kill_experiments.py
docs/                      # architecture.md, novelty_audit.md, red_team.md,
                           #   experiment_log.md
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
