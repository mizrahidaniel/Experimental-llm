"""byte_sampler: temperature, top-p, top-k + determinism."""

import torch

from sprl.inference.sampler import byte_sampler


def test_sampler_returns_int_in_range():
    logits = torch.randn(256)
    out = byte_sampler(logits, temperature=1.0, top_p=0.95)
    assert isinstance(out, int)
    assert 0 <= out < 256


def test_sampler_temperature_zero_argmax():
    logits = torch.zeros(256)
    logits[42] = 10.0
    out = byte_sampler(logits, temperature=0.0)
    assert out == 42


def test_sampler_deterministic_with_generator():
    torch.manual_seed(0)
    logits = torch.randn(256)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = byte_sampler(logits, temperature=1.0, top_p=0.95, generator=g1)
    b = byte_sampler(logits, temperature=1.0, top_p=0.95, generator=g2)
    assert a == b


def test_sampler_different_seeds_can_differ():
    torch.manual_seed(0)
    logits = torch.randn(256) * 0.5
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    samples1 = [byte_sampler(logits, temperature=1.0, top_p=0.95, generator=g1) for _ in range(10)]
    samples2 = [byte_sampler(logits, temperature=1.0, top_p=0.95, generator=g2) for _ in range(10)]
    # At least one element should differ.
    assert samples1 != samples2


def test_sampler_top_k_restricts_support():
    """With top_k=1, only the argmax can be sampled."""
    logits = torch.zeros(256)
    logits[42] = 10.0
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        assert byte_sampler(logits, temperature=1.0, top_p=1.0, top_k=1, generator=g) == 42


def test_sampler_top_p_restricts_to_nucleus():
    """With strongly-peaked logits and small top_p, only the top byte is in
    the nucleus."""
    logits = torch.full((256,), -10.0)
    logits[100] = 5.0
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        assert byte_sampler(logits, temperature=1.0, top_p=0.5, generator=g) == 100
