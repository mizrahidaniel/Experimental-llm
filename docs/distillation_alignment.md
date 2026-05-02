# Distillation alignment

Direct teacher-token KL works **only when student and teacher share an
identical tokenizer**. SPRL-v3.1 enforces this in `SPRLConfig.validate()`.

## Three modes

| `distill_mode` | Use when | KL surface |
|---|---|---|
| `teacher_token_kl` | Student vocab == teacher vocab (same tokenizer) | Student LM-head logits vs teacher logits |
| `auxiliary_teacher_token_head` | Tokenizers differ (e.g. BLT bytes vs Llama BPE) | A separate aux head's logits (in teacher vocab) vs teacher logits; main LM head trains via student-vocab CE only |
| `sequence_level` | You don't want to precompute or stream teacher logits | Kim & Rush 2016: student CE on teacher's *generated text*. No KL, no logit storage. |

## End-to-end recipe (no precompute step)

If you're not ready to spend $100-300 on a cloud GPU to precompute 6-12 TB
of teacher logits, sequence-level distillation is the pragmatic alternative.
Two scripts wrap the whole flow:

```bash
# 1. Sample teacher completions (one-time, run on whatever GPU you have).
#    Falls back to MarkovTeacher when transformers / GPU unavailable, so the
#    pipeline can be smoke-tested offline.
python scripts/generate_distill_corpus.py \
    --teacher meta-llama/Llama-3.1-8B-Instruct \
    --prompts data/seed_prompts.jsonl \
    --output  data/distill_corpus/llama_3_1_8b.jsonl \
    --num-samples 100000 \
    --max-new-tokens 512 \
    --temperature 0.8

# 2. Tokenize the resulting corpus with the teacher's tokenizer.
#    Resumable: re-running picks up where it left off.
python scripts/tokenize_corpus.py \
    --tokenizer llama_3_1_8b \
    --input  data/distill_corpus/llama_3_1_8b.jsonl \
    --output data/tokenized/llama_3_1_8b_distill \
    --field  completion \
    --shard-size 1000000

# 3. Train the student on the tokenized corpus with sequence_level distillation.
#    The student's CE head treats teacher-generated tokens as ground truth;
#    no KL term is added, so no teacher logits are needed at training time.
#    Set distill_mode=sequence_level so the validator doesn't demand
#    distill_store_teacher_logsumexp etc.
python scripts/run_full_pretrain.py \
    --variant recommended
```

Pair `tokenized_iterator(...)` from `sprl.data` with the existing
`train_one_step` to drive the loop:

```python
from sprl.data import tokenized_iterator

for batch in tokenized_iterator(
    "data/tokenized/llama_3_1_8b_distill",
    batch_size=cfg.training.batch_size,
    seq_len=cfg.training.seq_len_patches,
):
    # batch = {"token_ids": [B, S], "mtp_targets": [B, S, depth]}
    loss = train_one_step(model, batch, optim, cfg, step=step)
```

Quality vs token-level KL: ~50-70% per token (per MiniLLM ablations); zero
infrastructure overhead. Use this path for a first 5080 run; promote to
`teacher_token_kl` only if you've measured that distillation is the
bottleneck.

## Hard validation matrix

`cfg.validate()` errors on these combinations:

1. `patcher.enabled = true` (BLT) + `distill_mode = teacher_token_kl`:
   incoherent — 256-byte vocab cannot align with a ≥32K BPE teacher.
2. `distill_mode = teacher_token_kl` + non-BLT student vocab ≤ 1024: too
   small to align with any common teacher.
3. `distill_mode = teacher_token_kl` without `distill_store_teacher_logsumexp`:
   normalized KL with truncated top-K requires the per-token logsumexp.
4. `distill_kl_direction` not in `{teacher_forward_kl, reverse_kl}`:
   name the direction explicitly so debugging behaviour is unambiguous.

## Auxiliary teacher-token head

`sprl/heads/auxiliary_teacher_token_head.py` ships an
`AuxiliaryTeacherTokenHead` that sits on top of the student trunk and
predicts in the teacher's vocab. Its KL receives gradients into the trunk;
the main LM head still trains on student-vocab CE.

**v3.1 wiring**: `SPRLv2.__init__` instantiates the aux head automatically
when `cfg.training.distill_enabled and cfg.training.distill_mode ==
"auxiliary_teacher_token_head"`. The `forward_from_*` methods then attach
`out.aux_logits`, and `compute_total_loss` automatically routes the KL
against `out.aux_logits` (in teacher vocab) instead of against the main LM
head's logits. The aux head's `teacher_vocab_size` is looked up from
`sprl.tokenizer.vocab_size_for(cfg.training.distill_teacher_base)`.

```python
from sprl.heads import AuxiliaryTeacherTokenHead

aux = AuxiliaryTeacherTokenHead(d_model=768, teacher_vocab_size=128_000)
aux_logits = aux(trunk_hidden)  # [..., 128_000]
distill_loss = F.kl_div(F.log_softmax(aux_logits, -1),
                        F.softmax(teacher_logits, -1),
                        reduction="batchmean")
```

Optional weight-tying: pass `tied_embedding=teacher_input_embedding` if
you've loaded the teacher's input embedding (saves ~98M params at d=768,
vocab=128K).

## KL direction

| Direction | Formula | Behaviour |
|---|---|---|
| `teacher_forward_kl` (default) | KL(teacher ‖ student) | Mode-covering. Encourages student to put mass where teacher does. Standard distillation. |
| `reverse_kl` | KL(student ‖ teacher) | Mode-seeking. MiniLLM-style, picks one mode and concentrates there. |

```python
from sprl.training.distill import distillation_kl_loss

loss = distillation_kl_loss(student_logits, teacher_logits,
                            temperature=2.0,
                            direction="teacher_forward_kl")
```

## Phase-aware teacher selection

Per spec §6.4, use the **base** teacher for raw-web pretraining (Phase 3a)
to avoid RLHF style/alignment artifacts, and the **instruct** teacher for
synthetic / instruction phases (3c) so the student aligns behaviourally.

```python
from sprl.training.distill import select_teacher_for_phase

teacher_3a = select_teacher_for_phase(cfg, "3a")        # base
teacher_3c = select_teacher_for_phase(cfg, "synthetic") # instruct
```

Set `cfg.training.distill_teacher_base` and
`cfg.training.distill_teacher_instruct` in your YAML.

## Top-K logit storage

For 30B tokens at top-32, raw teacher logits with int32 indices + fp16
scores + fp32 logsumexp = ~196 bytes/token = ~5.9 TB. Plan accordingly:

- 6 TB external SSD or NAS mandatory for 30B run.
- 12 TB for 50B run.
- Compression options: index difference encoding (~30% saving),
  dropping tail-mass (small KL accuracy loss).

The shipped `sprl/training/teacher_logits.py` writes:

| Field | dtype | Size per token |
|---|---|---|
| `byte_targets` (or `targets`) | int32 | 4 B |
| `topk_indices` | int32 | 32 × 4 = 128 B |
| `topk_logits` | float16 | 32 × 2 = 64 B |
| `teacher_logsumexp` | float32 | 4 B (required for normalized KL) |
| `tail_mass` (optional) | float32 | 4 B |

Validation requires `distill_store_teacher_logsumexp = true` whenever
`distill_mode = teacher_token_kl`.
