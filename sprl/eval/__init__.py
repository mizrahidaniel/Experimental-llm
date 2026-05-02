"""Evaluation subpackage.

Each eval module exposes a single `evaluate(model, ...)` entry point that takes
an `SPRLv2` instance and returns a metric dict. Modules that wrap external
benchmarks (MMLU/GSM8K/HumanEval/GPQA) try `datasets` first and fall back to a
tiny in-process fixture so unit tests pass without network.
"""

from sprl.eval.byte_io import (  # noqa: F401
    bytes_to_patch_emb,
    score_continuation,
    generate_bytes,
    multiple_choice_score,
)
from sprl.eval.perplexity import evaluate_perplexity  # noqa: F401
from sprl.eval.mmlu import evaluate_mmlu  # noqa: F401
from sprl.eval.gsm8k import evaluate_gsm8k  # noqa: F401
from sprl.eval.humaneval import evaluate_humaneval  # noqa: F401
from sprl.eval.gpqa import evaluate_gpqa  # noqa: F401
from sprl.eval.ruler import evaluate_ruler  # noqa: F401
from sprl.eval.state_tracking import evaluate_s5_state_tracking  # noqa: F401
from sprl.eval.harness import SPRLHarnessAdapter  # noqa: F401
