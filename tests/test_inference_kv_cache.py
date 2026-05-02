"""KVCache for MLA inference: append, get, truncate, clear."""

import pytest
import torch

from sprl.inference.kv_cache import KVCache, MLAKVCache


def test_mla_kv_cache_append_and_get():
    cache = MLAKVCache(max_len=8, d_c=4, n_heads=2, d_r=2)
    c_KV = torch.randn(1, 3, 4)
    K_R = torch.randn(1, 3, 2 * 2)
    cache.append(c_KV, K_R)
    assert cache.length == 3
    c, k = cache.get()
    assert c.shape == (1, 3, 4)
    assert k.shape == (1, 3, 4)
    assert torch.allclose(c, c_KV)
    assert torch.allclose(k, K_R)


def test_mla_kv_cache_overflow_raises():
    cache = MLAKVCache(max_len=4, d_c=4, n_heads=2, d_r=2)
    big = torch.randn(1, 5, 4)
    big_kr = torch.randn(1, 5, 4)
    with pytest.raises(RuntimeError):
        cache.append(big, big_kr)


def test_mla_kv_cache_truncate_clear():
    cache = MLAKVCache(max_len=8, d_c=4, n_heads=2, d_r=2)
    cache.append(torch.randn(1, 5, 4), torch.randn(1, 5, 4))
    assert cache.length == 5
    cache.truncate(3)
    assert cache.length == 3
    cache.clear()
    assert cache.length == 0


def test_kv_cache_for_model():
    cache = KVCache.for_model(n_layers=4, max_len=16, d_c=8, n_heads=4, d_r=2)
    assert len(cache.per_layer) == 4
    assert cache.length == 0
    cache.per_layer[0].append(torch.randn(1, 2, 8), torch.randn(1, 2, 8))
    cache.per_layer[1].append(torch.randn(1, 2, 8), torch.randn(1, 2, 8))
    cache.truncate(0)
    for l in cache.per_layer:
        assert l.length == 0


def test_mla_kv_cache_append_multiple_chunks():
    cache = MLAKVCache(max_len=10, d_c=4, n_heads=2, d_r=2)
    cache.append(torch.randn(1, 3, 4), torch.randn(1, 3, 4))
    cache.append(torch.randn(1, 4, 4), torch.randn(1, 4, 4))
    assert cache.length == 7
    c, k = cache.get()
    assert c.shape == (1, 7, 4)
