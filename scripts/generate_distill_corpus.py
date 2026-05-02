"""Generate a sequence-level distillation corpus from a teacher model.

Sequence-level distillation (Kim & Rush 2016) trains the student on the
teacher's *sampled output* with regular cross-entropy — no logit storage,
no KL divergence. Lower distillation efficiency than token-level KL but
two orders of magnitude cheaper to set up: no 6-12 TB shard tree, no
custom KL kernel, just text in / text out.

This script samples N completions from a teacher LM. Output is a JSONL of
records `{"prompt": str | None, "completion": str}` that can be fed back
through `scripts/tokenize_corpus.py` to produce the training shards the
student consumes.

Tries `transformers.AutoModelForCausalLM.from_pretrained(...)`; if it can't
load the teacher (no transformers, no GPU, gated repo), falls back to a
toy MarkovTeacher that emits character-level Markov text. The fallback is
for unit tests and dry-runs only — real corpora require the real teacher.

Usage:
    # With a real teacher on a GPU:
    python scripts/generate_distill_corpus.py \\
        --teacher meta-llama/Llama-3.1-8B-Instruct \\
        --prompts data/seed_prompts.jsonl \\
        --output  data/distill_corpus/llama_3_1_8b_instruct.jsonl \\
        --max-new-tokens 512 \\
        --temperature 0.8 \\
        --num-samples 100000

    # Dry-run (Markov fallback, no network):
    python scripts/generate_distill_corpus.py \\
        --teacher mock \\
        --prompts data/fixtures/seed_prompts.jsonl \\
        --output  /tmp/mock_corpus.jsonl \\
        --num-samples 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, Optional


class MarkovTeacher:
    """Trivial offline fallback. Generates char-level Markov text seeded
    from the prompt. Quality is awful by design — present so unit tests
    and dry-runs work without a real LM."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._alphabet = (
            "abcdefghijklmnopqrstuvwxyz "
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ".,;:!?'\"()-\n"
        )

    def generate(self, prompt: str, max_new_chars: int = 200) -> str:
        out = []
        prev = prompt[-1] if prompt else " "
        for _ in range(max_new_chars):
            ch = self.rng.choice(self._alphabet)
            # Make spaces twice as likely after a space (looks slightly more wordlike).
            if prev == " ":
                ch = self.rng.choice(self._alphabet + " ")
            out.append(ch)
            prev = ch
        return "".join(out)


def iter_prompts(path: Optional[Path], n_total: int) -> Iterator[str]:
    """Yield up to n_total prompts. If `path` is None, yield empty strings
    (unconditional generation)."""
    if path is None:
        for _ in range(n_total):
            yield ""
        return
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_total:
                return
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line  # treat raw line as prompt
                continue
            text = obj.get("prompt") or obj.get("text") or ""
            yield text


def load_teacher(name: str):
    """Returns either an HF model+tokenizer pair or a MarkovTeacher fallback."""
    if name == "mock":
        return ("mock", MarkovTeacher())
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        print(f"[generate] transformers not installed; falling back to MarkovTeacher")
        return ("mock", MarkovTeacher())
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")
    except Exception as e:
        print(f"[generate] failed to load {name!r} ({e}); falling back to MarkovTeacher")
        return ("mock", MarkovTeacher())
    mdl.eval()
    return ("hf", (mdl, tok))


def generate_with_hf(model, tokenizer, prompt: str, max_new_tokens: int,
                     temperature: float, top_p: float) -> str:
    import torch  # local import — only needed on the HF path

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    n_in = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0, n_in:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True,
                   help="HF model id or 'mock' for the offline Markov fallback")
    p.add_argument("--prompts", type=Path, default=None,
                   help="JSONL with seed prompts (field 'prompt' or 'text'); omit for unconditional")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend, payload = load_teacher(args.teacher)
    print(f"[generate] backend={backend} teacher={args.teacher} → {args.output}")

    written = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for prompt in iter_prompts(args.prompts, args.num_samples):
            if backend == "hf":
                model, tokenizer = payload
                completion = generate_with_hf(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            else:
                completion = payload.generate(prompt, max_new_chars=args.max_new_tokens)
            rec = {"prompt": prompt, "completion": completion}
            out_f.write(json.dumps(rec) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"[generate] wrote {written} samples")

    print(f"[generate] done: {written} samples → {args.output}")


if __name__ == "__main__":
    main()
