"""GPQA-diamond multiple-choice evaluator.

Same shape as MMLU but harder distractors. The HF dataset is gated; we attempt
`datasets.load_dataset('Idavidrein/gpqa', 'gpqa_diamond')` and fall back to a
small in-process fixture when the dataset is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sprl.eval.byte_io import multiple_choice_score
from sprl.model import SPRLv2


# ---------------------------------------------------------------------------
# Fixture: 8 questions in graduate-physics flavor. Plumbing test only.
# ---------------------------------------------------------------------------

_FIXTURE: List[dict] = [
    {
        "question": "Which particle mediates the strong nuclear force?",
        "choices": ["photon", "gluon", "W boson", "graviton"],
        "answer": 1,
    },
    {
        "question": "Spin of the photon?",
        "choices": ["0", "1/2", "1", "2"],
        "answer": 2,
    },
    {
        "question": "Which is a non-Abelian gauge group of the Standard Model?",
        "choices": ["U(1)", "SU(2)", "Z_2", "R"],
        "answer": 1,
    },
    {
        "question": "Heisenberg uncertainty: Δx Δp >= ?",
        "choices": ["0", "h", "ħ", "ħ/2"],
        "answer": 3,
    },
    {
        "question": "Hawking radiation temperature scales as 1/?",
        "choices": ["mass", "mass^2", "log mass", "sqrt mass"],
        "answer": 0,
    },
    {
        "question": "Which symmetry implies energy conservation?",
        "choices": ["spatial translation", "rotation", "time translation", "gauge"],
        "answer": 2,
    },
    {
        "question": "QED running coupling at large energy:",
        "choices": ["increases", "decreases", "constant", "oscillates"],
        "answer": 0,
    },
    {
        "question": "Pauli exclusion applies to which particles?",
        "choices": ["bosons", "fermions", "all", "none"],
        "answer": 1,
    },
]

_LETTERS = ["A", "B", "C", "D"]


@dataclass
class GPQAConfig:
    n_shots: int = 0  # GPQA is typically zero-shot; can override
    max_questions: Optional[int] = None
    use_hf: bool = True
    seed: int = 0


def _format(q: dict) -> str:
    parts = [q["question"]]
    for letter, choice in zip(_LETTERS, q["choices"]):
        parts.append(f"{letter}. {choice}")
    parts.append("Answer:")
    return "\n".join(parts)


def _try_load_hf() -> Optional[List[dict]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception:
        return None
    out: List[dict] = []
    import random
    rng = random.Random(0)
    for row in ds:
        try:
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            order = list(range(4))
            rng.shuffle(order)
            ans_idx = order.index(0)
            out.append(
                {
                    "question": row["Question"],
                    "choices": [choices[i] for i in order],
                    "answer": ans_idx,
                }
            )
        except (KeyError, TypeError):
            continue
    return out


def evaluate_gpqa(model: SPRLv2, config: Optional[GPQAConfig] = None) -> Dict[str, float]:
    cfg = config or GPQAConfig()
    model.eval()

    questions: Optional[List[dict]] = None
    if cfg.use_hf:
        questions = _try_load_hf()
    if not questions:
        questions = list(_FIXTURE)
    if cfg.max_questions is not None:
        questions = questions[: cfg.max_questions]

    import random
    rng = random.Random(cfg.seed)

    n_correct = 0
    for i, q in enumerate(questions):
        pool = questions[:i] + questions[i + 1 :]
        shots = []
        if cfg.n_shots > 0 and len(pool) >= cfg.n_shots:
            shots = rng.sample(pool, cfg.n_shots)
        prefix = ""
        for s in shots:
            prefix += _format(s) + " " + _LETTERS[s["answer"]] + "\n\n"
        prompt = prefix + _format(q) + " "
        choices = [_LETTERS[j] for j in range(len(q["choices"]))]
        pred = multiple_choice_score(model, prompt, choices, normalize_by_length=False)
        if pred == q["answer"]:
            n_correct += 1

    return {
        "accuracy": n_correct / max(len(questions), 1),
        "n_questions": float(len(questions)),
        "n_correct": float(n_correct),
    }
