"""MMLU eval — fixture path."""

from sprl.eval.mmlu import MMLUConfig, evaluate_mmlu

from tests._eval_helpers import tiny_model


def test_mmlu_fixture_runs():
    model = tiny_model()
    m = evaluate_mmlu(
        model,
        MMLUConfig(n_shots=2, max_questions=4, use_hf=False),
    )
    assert "accuracy" in m
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["n_questions"] == 4


def test_mmlu_per_subject_keys():
    model = tiny_model()
    m = evaluate_mmlu(
        model,
        MMLUConfig(n_shots=0, max_questions=20, use_hf=False),
    )
    # Fixture has 4 subjects (5 questions each).
    subj_keys = [k for k in m if k.startswith("acc_")]
    assert len(subj_keys) == 4
    for k in subj_keys:
        assert 0.0 <= m[k] <= 1.0


def test_mmlu_zero_shot_does_not_crash():
    model = tiny_model()
    m = evaluate_mmlu(
        model,
        MMLUConfig(n_shots=0, max_questions=2, use_hf=False),
    )
    assert m["n_questions"] == 2
