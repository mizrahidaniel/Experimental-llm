"""HumanEval pass@1 with subprocess-sandboxed Python execution.

Each problem provides a function signature + docstring. The model generates a
completion; we paste it under the prompt, append the test harness, and run
the resulting file in a fresh `python3 -c …` subprocess with a wall-clock
timeout. Pass@1 is the fraction of problems whose harness exits 0.

Subprocess isolation is a *minimum* sandbox — it does not prevent filesystem
access. For real evaluation use a containerized harness (e.g. the official
`human-eval` repo). We mirror its API surface so swapping is trivial.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from sprl.eval.byte_io import generate_bytes
from sprl.model import SPRLv2


# ---------------------------------------------------------------------------
# Tiny built-in fixture: 3 trivial problems. Tests the harness, not capability.
# ---------------------------------------------------------------------------

_FIXTURE: List[dict] = [
    {
        "task_id": "fixture/0",
        "prompt": (
            "def add(a, b):\n"
            "    \"\"\"Return a + b.\"\"\"\n"
        ),
        "canonical_solution": "    return a + b\n",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(-1, 1) == 0\n"
            "    assert candidate(0, 0) == 0\n"
        ),
        "entry_point": "add",
    },
    {
        "task_id": "fixture/1",
        "prompt": (
            "def is_even(n):\n"
            "    \"\"\"Return True iff n is even.\"\"\"\n"
        ),
        "canonical_solution": "    return n % 2 == 0\n",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(2) is True\n"
            "    assert candidate(3) is False\n"
            "    assert candidate(0) is True\n"
        ),
        "entry_point": "is_even",
    },
    {
        "task_id": "fixture/2",
        "prompt": (
            "def double(x):\n"
            "    \"\"\"Return x doubled.\"\"\"\n"
        ),
        "canonical_solution": "    return x * 2\n",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(3) == 6\n"
            "    assert candidate(-2) == -4\n"
        ),
        "entry_point": "double",
    },
]


# ---------------------------------------------------------------------------


@dataclass
class HumanEvalConfig:
    timeout_s: float = 5.0
    max_new_bytes: int = 256
    use_hf: bool = True
    max_problems: Optional[int] = None
    use_canonical: bool = False  # if True, run canonical solutions (sanity)


def _try_load_hf() -> Optional[List[dict]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset("openai_humaneval", split="test")
    except Exception:
        return None
    out: List[dict] = []
    for row in ds:
        try:
            out.append(
                {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "canonical_solution": row["canonical_solution"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                }
            )
        except (KeyError, TypeError):
            continue
    return out


def _truncate_completion(completion: str) -> str:
    """Trim at the first dedented line (start of next def/class)."""
    lines = completion.splitlines()
    keep: List[str] = []
    for line in lines:
        if line.strip() == "":
            keep.append(line)
            continue
        # First non-empty line determines indent expectation.
        if not line.startswith((" ", "\t")) and any(s.strip() for s in keep):
            break
        keep.append(line)
    return "\n".join(keep) + ("\n" if keep else "")


def _run_in_subprocess(code: str, timeout_s: float) -> tuple[bool, str]:
    """Run `code` in a fresh interpreter; return (passed, stderr_or_msg)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(code)
        path = tf.name
    try:
        # Disable bytecode + spawn a clean environment.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            r = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        ok = r.returncode == 0
        return ok, (r.stderr or r.stdout)[-512:]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _assemble_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    return (
        prompt
        + completion
        + "\n\n"
        + test
        + f"\n\ncheck({entry_point})\n"
    )


def evaluate_humaneval(
    model: Optional[SPRLv2],
    config: Optional[HumanEvalConfig] = None,
) -> Dict[str, float]:
    """Pass@1 on HumanEval. If model is None, runs canonical solutions
    (useful as a self-test of the harness).
    """
    cfg = config or HumanEvalConfig()
    problems: Optional[List[dict]] = None
    if cfg.use_hf:
        problems = _try_load_hf()
    if not problems:
        problems = list(_FIXTURE)
    if cfg.max_problems is not None:
        problems = problems[: cfg.max_problems]

    use_canonical = cfg.use_canonical or model is None
    if model is not None:
        model.eval()

    n_pass = 0
    failures: List[str] = []
    for prob in problems:
        if use_canonical:
            completion = prob["canonical_solution"]
        else:
            assert model is not None
            gen = generate_bytes(
                model, prob["prompt"], max_new_bytes=cfg.max_new_bytes
            )
            completion = _truncate_completion(gen)
        program = _assemble_program(
            prob["prompt"], completion, prob["test"], prob["entry_point"]
        )
        ok, msg = _run_in_subprocess(program, cfg.timeout_s)
        if ok:
            n_pass += 1
        else:
            failures.append(f"{prob['task_id']}: {msg.splitlines()[-1] if msg else ''}")

    return {
        "pass@1": n_pass / max(len(problems), 1),
        "n_problems": float(len(problems)),
        "n_passed": float(n_pass),
        "n_failed": float(len(problems) - n_pass),
        # Don't return failures by default — keep dict numeric for logging.
    }
