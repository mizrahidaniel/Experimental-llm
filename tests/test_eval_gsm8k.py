"""GSM8K eval — answer extraction + fixture path."""

from sprl.eval.gsm8k import GSM8KConfig, evaluate_gsm8k, extract_answer

from tests._eval_helpers import tiny_model


def test_extract_answer_marker():
    assert extract_answer("Reasoning. #### 42") == 42.0
    assert extract_answer("step. step. #### -3.5") == -3.5


def test_extract_answer_falls_back_to_last_number():
    assert extract_answer("the result is 7") == 7.0
    assert extract_answer("nothing here") is None


def test_extract_answer_prefers_marker_over_others():
    # If both '####' marker and stray numbers, marker wins.
    assert extract_answer("we add 1 + 2 to get something. #### 99") == 99.0


def test_gsm8k_fixture_runs():
    model = tiny_model()
    m = evaluate_gsm8k(
        model,
        GSM8KConfig(n_shots=2, max_questions=2, max_new_bytes=24, use_hf=False),
    )
    assert "accuracy" in m
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["n_questions"] == 2
    assert "extraction_rate" in m
