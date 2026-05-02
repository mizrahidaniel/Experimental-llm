"""S5 state-tracking eval (kill-criterion diagnostic for Bet C)."""

from sprl.eval.state_tracking import (
    S5Config,
    _compose,
    _format_perm,
    _parse_perm,
    evaluate_s5_state_tracking,
)

from tests._eval_helpers import tiny_model


def test_compose_identity():
    p = (0, 1, 2, 3, 4)
    q = (4, 3, 2, 1, 0)
    assert _compose(p, q) == q
    assert _compose(q, p) == q


def test_compose_associativity():
    a = (1, 0, 3, 2, 4)
    b = (2, 4, 1, 0, 3)
    c = (3, 0, 4, 1, 2)
    left = _compose(_compose(a, b), c)
    right = _compose(a, _compose(b, c))
    assert left == right


def test_format_and_parse_perm_round_trip():
    p = (4, 2, 0, 3, 1)
    s = _format_perm(p)
    assert s == "53142"
    assert _parse_perm(s) == p
    assert _parse_perm("X 5 3 1 4 2 ignore") == p


def test_parse_perm_rejects_non_permutation():
    assert _parse_perm("11111") is None
    assert _parse_perm("abc") is None


def test_s5_score_method_runs():
    model = tiny_model()
    m = evaluate_s5_state_tracking(
        model,
        S5Config(seq_lengths=[3], n_per_length=2, method="score"),
    )
    assert "accuracy" in m
    assert "acc_L3" in m
    assert m["chance"] == 1.0 / 120.0
    assert m["n_examples"] == 2


def test_s5_generate_method_runs():
    model = tiny_model()
    m = evaluate_s5_state_tracking(
        model,
        S5Config(seq_lengths=[2], n_per_length=2, method="generate", max_new_bytes=8),
    )
    assert "accuracy" in m
