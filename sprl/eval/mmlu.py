"""MMLU-style 5-shot evaluator.

Loads from HuggingFace `datasets` if available; otherwise uses a tiny built-in
20-question fixture so unit tests run on CPU with no network. Scoring is
log-prob over the answer letter ("A"/"B"/"C"/"D"), the standard MMLU
convention. Returns per-subject and overall accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sprl.eval.byte_io import multiple_choice_score
from sprl.model import SPRLv2


# ---------------------------------------------------------------------------
# Fixture: 20 questions over 4 subjects (5 each), CPU-OK.
# Trivial-knowledge style — checks the eval plumbing, not real performance.
# ---------------------------------------------------------------------------

_FIXTURE: List[dict] = [
    # --- elementary_math (5) -----------------------------------------------
    {"subject": "elementary_math", "question": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "answer": 1},
    {"subject": "elementary_math", "question": "What is 7 * 6?", "choices": ["36", "40", "42", "48"], "answer": 2},
    {"subject": "elementary_math", "question": "What is 12 / 4?", "choices": ["2", "3", "4", "6"], "answer": 1},
    {"subject": "elementary_math", "question": "What is 9 - 5?", "choices": ["2", "3", "4", "5"], "answer": 2},
    {"subject": "elementary_math", "question": "What is 10 + 15?", "choices": ["20", "23", "25", "30"], "answer": 2},
    # --- world_facts (5) ---------------------------------------------------
    {"subject": "world_facts", "question": "What is the capital of France?", "choices": ["Berlin", "Paris", "Rome", "Madrid"], "answer": 1},
    {"subject": "world_facts", "question": "Which planet is closest to the Sun?", "choices": ["Venus", "Earth", "Mercury", "Mars"], "answer": 2},
    {"subject": "world_facts", "question": "What is the largest ocean?", "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": 3},
    {"subject": "world_facts", "question": "Which element has the symbol 'O'?", "choices": ["Gold", "Oxygen", "Osmium", "Iron"], "answer": 1},
    {"subject": "world_facts", "question": "How many continents are there?", "choices": ["5", "6", "7", "8"], "answer": 2},
    # --- grammar (5) -------------------------------------------------------
    {"subject": "grammar", "question": "Which is a noun?", "choices": ["run", "quickly", "table", "happy"], "answer": 2},
    {"subject": "grammar", "question": "Which is a verb?", "choices": ["jump", "blue", "small", "house"], "answer": 0},
    {"subject": "grammar", "question": "Which is plural?", "choices": ["dog", "cat", "mice", "bird"], "answer": 2},
    {"subject": "grammar", "question": "Which is an adjective?", "choices": ["sing", "loudly", "tall", "house"], "answer": 2},
    {"subject": "grammar", "question": "Which is an adverb?", "choices": ["beautiful", "quickly", "tree", "stone"], "answer": 1},
    # --- logic (5) ---------------------------------------------------------
    {"subject": "logic", "question": "If A>B and B>C then A?C.", "choices": ["<", "=", ">", "?"], "answer": 2},
    {"subject": "logic", "question": "Negation of 'all are X'?", "choices": ["all are not X", "some are not X", "none are X", "X is some"], "answer": 1},
    {"subject": "logic", "question": "A implies B; A is true. Then B is?", "choices": ["false", "true", "unknown", "both"], "answer": 1},
    {"subject": "logic", "question": "Which is a tautology?", "choices": ["P and not P", "P or not P", "P", "not P"], "answer": 1},
    {"subject": "logic", "question": "If 'no X is Y', is 'some X is Y'?", "choices": ["true", "false", "unknown", "both"], "answer": 1},
]

_LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------


@dataclass
class MMLUConfig:
    n_shots: int = 5
    max_questions: Optional[int] = None
    use_hf: bool = True               # try HF datasets if available
    hf_subset: Optional[str] = None   # e.g. "high_school_biology"; None = all
    seed: int = 0


def _format_question(q: dict) -> str:
    """Render a question + four choices, ending with 'Answer:'."""
    parts = [q["question"]]
    for letter, choice in zip(_LETTERS, q["choices"]):
        parts.append(f"{letter}. {choice}")
    parts.append("Answer:")
    return "\n".join(parts)


def _build_few_shot_prefix(shots: List[dict]) -> str:
    if not shots:
        return ""
    rendered = []
    for q in shots:
        rendered.append(_format_question(q) + " " + _LETTERS[q["answer"]])
    return "\n\n".join(rendered) + "\n\n"


def _try_load_hf(subset: Optional[str], seed: int) -> Optional[List[dict]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        if subset:
            ds = load_dataset("cais/mmlu", subset, split="test")
        else:
            ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        return None
    out: List[dict] = []
    for row in ds:
        try:
            out.append(
                {
                    "subject": row.get("subject", "unknown"),
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": int(row["answer"]),
                }
            )
        except (KeyError, TypeError):
            continue
    return out


def evaluate_mmlu(model: SPRLv2, config: Optional[MMLUConfig] = None) -> Dict[str, float]:
    """Evaluate MMLU-style multiple choice. Returns per-subject + overall accuracy."""
    cfg = config or MMLUConfig()
    model.eval()

    questions: Optional[List[dict]] = None
    if cfg.use_hf:
        questions = _try_load_hf(cfg.hf_subset, cfg.seed)
    if not questions:
        questions = list(_FIXTURE)

    if cfg.max_questions is not None:
        questions = questions[: cfg.max_questions]

    # Group by subject so few-shot prefix samples from the same subject when possible.
    by_subject: Dict[str, List[dict]] = {}
    for q in questions:
        by_subject.setdefault(q["subject"], []).append(q)

    correct: Dict[str, int] = {}
    total: Dict[str, int] = {}

    import random
    rng = random.Random(cfg.seed)

    for subject, qs in by_subject.items():
        for i, q in enumerate(qs):
            # Few-shot pool excludes the held-out question itself.
            pool = qs[:i] + qs[i + 1 :]
            if len(pool) < cfg.n_shots:
                shots = pool
            else:
                shots = rng.sample(pool, cfg.n_shots)
            prompt = _build_few_shot_prefix(shots) + _format_question(q) + " "
            choices = [_LETTERS[j] for j in range(len(q["choices"]))]
            pred = multiple_choice_score(model, prompt, choices, normalize_by_length=False)
            ok = int(pred == q["answer"])
            correct[subject] = correct.get(subject, 0) + ok
            total[subject] = total.get(subject, 0) + 1

    out: Dict[str, float] = {}
    n_total = 0
    n_correct = 0
    for subject in by_subject:
        c, t = correct.get(subject, 0), total.get(subject, 0)
        out[f"acc_{subject}"] = c / max(t, 1)
        n_correct += c
        n_total += t
    out["accuracy"] = n_correct / max(n_total, 1)
    out["n_questions"] = float(n_total)
    out["n_subjects"] = float(len(by_subject))
    return out
