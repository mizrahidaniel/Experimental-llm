"""Decode-speed benchmark with MTP speculation."""

from sprl.bench.decode_speed import DecodeSpeedConfig, benchmark_decode_speed

from tests._eval_helpers import tiny_cfg
from sprl.model import SPRLv2


def test_decode_speed_mtp_enabled():
    cfg = tiny_cfg()
    cfg.mtp.enabled = True
    model = SPRLv2(cfg).eval()
    m = benchmark_decode_speed(
        model,
        DecodeSpeedConfig(prompt="hi", max_new_bytes=4, warmup=0, n_runs=1),
    )
    assert m["baseline_tokens_per_sec"] > 0
    assert m["mtp_tokens_per_sec"] > 0
    assert 0.0 <= m["acceptance_rate"] <= 1.0
    assert m["n_new_bytes"] == 4


def test_decode_speed_mtp_disabled_falls_back_gracefully():
    cfg = tiny_cfg()
    cfg.mtp.enabled = False
    model = SPRLv2(cfg).eval()
    m = benchmark_decode_speed(
        model,
        DecodeSpeedConfig(prompt="hi", max_new_bytes=4, warmup=0, n_runs=1),
    )
    assert m["baseline_tokens_per_sec"] > 0
    # When MTP off, mtp_tokens_per_sec is 0 and speedup is 1.0.
    assert m["mtp_tokens_per_sec"] == 0.0
    assert m["speedup"] == 1.0
