"""RULER synthetic long-context eval."""

from sprl.eval.ruler import RULERConfig, evaluate_ruler

from tests._eval_helpers import tiny_model


def test_ruler_runs_with_tiny_lengths():
    model = tiny_model()
    m = evaluate_ruler(
        model,
        RULERConfig(
            lengths=[64, 128],
            n_per_length=2,
            tasks=["niah_single", "kv_recall"],
            max_new_bytes=8,
        ),
    )
    assert "accuracy" in m
    # 2 lengths * 2 tasks * 2 examples = 8
    assert m["n_examples"] == 8
    for k in (
        "acc_niah_single_L64",
        "acc_niah_single_L128",
        "acc_kv_recall_L64",
        "acc_kv_recall_L128",
    ):
        assert k in m
        assert 0.0 <= m[k] <= 1.0


def test_ruler_only_one_task():
    model = tiny_model()
    m = evaluate_ruler(
        model,
        RULERConfig(
            lengths=[64], n_per_length=1, tasks=["niah_single"], max_new_bytes=4
        ),
    )
    assert "acc_niah_single_L64" in m
    assert m["n_examples"] == 1
