"""lm-eval-harness adapter."""

from sprl.eval.harness import HarnessConfig, SPRLHarnessAdapter

from tests._eval_helpers import tiny_model


def test_harness_loglikelihood_returns_pairs():
    model = tiny_model()
    ad = SPRLHarnessAdapter(model)
    out = ad.loglikelihood([("hello ", "world"), ("foo ", "bar")])
    assert len(out) == 2
    for logp, is_greedy in out:
        assert isinstance(logp, float)
        assert isinstance(is_greedy, bool)


def test_harness_loglikelihood_rolling():
    model = tiny_model()
    ad = SPRLHarnessAdapter(model)
    out = ad.loglikelihood_rolling([("hello",), ("world",)])
    assert len(out) == 2
    assert all(isinstance(x, float) for x in out)


def test_harness_generate_until():
    model = tiny_model()
    ad = SPRLHarnessAdapter(model)
    gens = ad.generate_until(
        [("hi", {"until": "\n", "max_gen_toks": 4})]
    )
    assert len(gens) == 1
    assert isinstance(gens[0], str)
    # Generated up to 4 raw bytes (decode-replace may inflate when re-encoded;
    # the invariant we care about is "generation respects the byte budget").
    # Don't re-encode — just bound the visible character count loosely.
    assert len(gens[0]) <= 4


def test_harness_max_length_property():
    model = tiny_model()
    ad = SPRLHarnessAdapter(model, HarnessConfig(max_gen_bytes=16))
    assert ad.max_length == model.cfg.max_seq_len_patches * model.cfg.patcher.max_patch_bytes
    assert ad.device == "cpu"


def test_harness_accepts_instance_like_objects():
    model = tiny_model()
    ad = SPRLHarnessAdapter(model)

    class Inst:
        def __init__(self, *args):
            self.args = args

    out = ad.loglikelihood([Inst("a", "b"), Inst("c", "d")])
    assert len(out) == 2
