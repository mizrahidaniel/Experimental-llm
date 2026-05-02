"""GSM8K 8-shot chain-of-thought eval.

Each problem is rendered as a prompt with 8 worked examples, then the model is
asked to generate a CoT terminating in "#### <answer>". A regex extracts the
final numeric answer; correctness is exact-match on the integer/decimal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from sprl.eval.byte_io import generate_bytes
from sprl.model import SPRLv2


# ---------------------------------------------------------------------------
# Tiny built-in fixture (8 problems for the few-shot prefix + 4 test problems).
# Numbers are easy by design — these are correctness-of-eval-pipeline tests,
# not capability tests. The model is not expected to score above chance here.
# ---------------------------------------------------------------------------

_FEWSHOT: List[dict] = [
    {
        "question": "Janet has 3 apples. She buys 5 more. How many does she have?",
        "answer": "She starts with 3. She adds 5. 3 + 5 = 8.\n#### 8",
    },
    {
        "question": "A pencil costs $2. How much do 4 pencils cost?",
        "answer": "Each is $2. 4 * 2 = 8.\n#### 8",
    },
    {
        "question": "Tom has 12 marbles. He gives 4 away. How many remain?",
        "answer": "12 - 4 = 8.\n#### 8",
    },
    {
        "question": "A box has 6 rows of 5 eggs. How many eggs total?",
        "answer": "6 * 5 = 30.\n#### 30",
    },
    {
        "question": "If 18 cookies are split among 6 kids equally, how many each?",
        "answer": "18 / 6 = 3.\n#### 3",
    },
    {
        "question": "Sara is 7. Her brother is twice her age. How old is he?",
        "answer": "2 * 7 = 14.\n#### 14",
    },
    {
        "question": "A car travels 60 km/h for 2 hours. How far?",
        "answer": "60 * 2 = 120.\n#### 120",
    },
    {
        "question": "If a shirt costs $15 and is 50% off, what is the sale price?",
        "answer": "15 * 0.5 = 7.5.\n#### 7.5",
    },
]

_FIXTURE: List[dict] = [
    {
        "question": "Alice has 5 books. Bob gives her 3 more. How many books does Alice have now?",
        "answer": "5 + 3 = 8.\n#### 8",
    },
    {
        "question": "A pizza is cut into 8 slices. If 3 are eaten, how many remain?",
        "answer": "8 - 3 = 5.\n#### 5",
    },
    {
        "question": "There are 4 boxes with 9 candies each. How many candies total?",
        "answer": "4 * 9 = 36.\n#### 36",
    },
    {
        "question": "If 24 students are split into groups of 6, how many groups are there?",
        "answer": "24 / 6 = 4.\n#### 4",
    },
]


# ---------------------------------------------------------------------------


_ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_LAST_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str) -> Optional[float]:
    """Return the numeric value following '####', else the last number in text."""
    m = _ANSWER_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    nums = _LAST_NUM_RE.findall(text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


@dataclass
class GSM8KConfig:
    n_shots: int = 8
    max_questions: Optional[int] = None
    max_new_bytes: int = 96
    use_hf: bool = True
    seed: int = 0


def _build_prompt(shots: List[dict], q: dict) -> str:
    parts = []
    for s in shots:
        parts.append(f"Question: {s['question']}\nAnswer: {s['answer']}")
    parts.append(f"Question: {q['question']}\nAnswer:")
    return "\n\n".join(parts)


def _try_load_hf(seed: int) -> Optional[List[dict]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except Exception:
        return None
    out: List[dict] = []
    for row in ds:
        try:
            out.append({"question": row["question"], "answer": row["answer"]})
        except (KeyError, TypeError):
            continue
    return out


def evaluate_gsm8k(model: SPRLv2, config: Optional[GSM8KConfig] = None) -> Dict[str, float]:
    cfg = config or GSM8KConfig()
    model.eval()

    questions: Optional[List[dict]] = None
    if cfg.use_hf:
        questions = _try_load_hf(cfg.seed)
    if not questions:
        questions = list(_FIXTURE)

    if cfg.max_questions is not None:
        questions = questions[: cfg.max_questions]

    shots = _FEWSHOT[: cfg.n_shots]
    n_correct = 0
    n_total = 0
    n_extracted = 0

    for q in questions:
        prompt = _build_prompt(shots, q)
        gen = generate_bytes(model, prompt, max_new_bytes=cfg.max_new_bytes, stop="\n\n")
        pred = extract_answer(gen)
        gold = extract_answer(q["answer"])
        if pred is not None:
            n_extracted += 1
            if gold is not None and abs(pred - gold) < 1e-4:
                n_correct += 1
        n_total += 1

    return {
        "accuracy": n_correct / max(n_total, 1),
        "extraction_rate": n_extracted / max(n_total, 1),
        "n_questions": float(n_total),
        "n_correct": float(n_correct),
    }
