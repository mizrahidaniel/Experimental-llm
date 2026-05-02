"""HumanEval pass@1 — fixture path with canonical solutions."""

from sprl.eval.humaneval import HumanEvalConfig, evaluate_humaneval, _truncate_completion

from tests._eval_helpers import tiny_model


def test_humaneval_canonical_passes_all_fixture():
    # No model: harness runs canonical solutions; should be 100%.
    m = evaluate_humaneval(
        None, HumanEvalConfig(use_hf=False, use_canonical=True, timeout_s=5.0)
    )
    assert m["pass@1"] == 1.0
    assert m["n_problems"] == 3
    assert m["n_passed"] == 3


def test_humaneval_with_model_runs():
    # The tiny untrained model will fail every problem, but the harness must
    # complete and return numeric metrics without crashing.
    model = tiny_model()
    m = evaluate_humaneval(
        model,
        HumanEvalConfig(
            use_hf=False, use_canonical=False, timeout_s=5.0, max_new_bytes=64
        ),
    )
    assert "pass@1" in m
    assert 0.0 <= m["pass@1"] <= 1.0
    assert m["n_problems"] == 3


def test_truncate_completion():
    # First indented block is kept; subsequent dedented def/class is dropped.
    src = "    return x + 1\n\ndef other():\n    pass\n"
    out = _truncate_completion(src)
    assert "return x + 1" in out
    assert "def other" not in out
