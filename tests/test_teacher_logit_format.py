"""Round-trip: pack -> write_shard -> TeacherLogitDataset -> match within FP16 quant."""

from __future__ import annotations

import numpy as np
import torch

from sprl.training.teacher_logits import (
    TeacherLogitDataset,
    TeacherLogitManifest,
    pack_topk_logits,
    write_manifest,
    write_shard,
)


def _make_logits(N: int = 1024, V: int = 256, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, V, generator=g) * 5.0


def test_pack_topk_round_trip():
    V = 256
    K = 32
    logits = _make_logits(N=2048, V=V)
    idx, log = pack_topk_logits(logits, top_k=K)
    assert idx.shape == (2048, K)
    assert log.shape == (2048, K)
    assert idx.dtype == torch.int32
    assert log.dtype == torch.float16
    dense_top_vals, dense_top_idx = torch.topk(logits, K, dim=-1)
    assert torch.equal(idx.long(), dense_top_idx)
    err = (log.float() - dense_top_vals).abs().max().item()
    assert err < 1e-2, f"FP16 round-trip error too high: {err}"


def test_dataset_iteration_in_order(tmp_path):
    V = 256
    K = 8
    shard_size = 256
    n_shards = 3
    logits = _make_logits(N=shard_size * n_shards, V=V, seed=1)
    targets = torch.randint(0, V, (shard_size * n_shards,))
    for cid in range(n_shards):
        s, e = cid * shard_size, (cid + 1) * shard_size
        idx, lg = pack_topk_logits(logits[s:e], top_k=K)
        write_shard(
            tmp_path,
            chunk_id=cid,
            token_offset=s,
            byte_targets=targets[s:e].numpy().astype(np.int32),
            topk_indices=idx.numpy(),
            topk_logits=lg.numpy(),
        )
    write_manifest(
        tmp_path,
        TeacherLogitManifest(
            vocab_size=V, top_k=K, shard_size=shard_size,
            n_shards=n_shards, total_tokens=shard_size * n_shards,
        ),
    )
    ds = TeacherLogitDataset(tmp_path, prefetch=2)
    seen_offsets = []
    total_targets = []
    for shard in ds:
        seen_offsets.append(shard.token_offset)
        total_targets.append(shard.byte_targets)
    assert seen_offsets == [0, shard_size, 2 * shard_size]
    assert torch.cat(total_targets).tolist() == targets.tolist()


def test_dataset_seek_and_random_access(tmp_path):
    V = 256
    K = 4
    shard_size = 64
    n_shards = 4
    logits = _make_logits(N=shard_size * n_shards, V=V, seed=2)
    targets = torch.randint(0, V, (shard_size * n_shards,))
    for cid in range(n_shards):
        s, e = cid * shard_size, (cid + 1) * shard_size
        idx, lg = pack_topk_logits(logits[s:e], top_k=K)
        write_shard(
            tmp_path,
            chunk_id=cid,
            token_offset=s,
            byte_targets=targets[s:e].numpy().astype(np.int32),
            topk_indices=idx.numpy(),
            topk_logits=lg.numpy(),
        )
    write_manifest(
        tmp_path,
        TeacherLogitManifest(
            vocab_size=V, top_k=K, shard_size=shard_size,
            n_shards=n_shards, total_tokens=shard_size * n_shards,
        ),
    )

    ds = TeacherLogitDataset(tmp_path)
    shard = ds.get_by_offset(chunk_id=2, token_offset=130)
    assert shard.chunk_id == 2
    assert shard.token_offset == 130
    assert shard.byte_targets.shape[0] == shard_size - 2

    ds.seek(chunk_id=2, token_offset=130)
    first = next(iter(ds))
    assert first.chunk_id == 2
    assert first.token_offset == 130


def test_manifest_round_trip():
    m = TeacherLogitManifest(
        vocab_size=256, top_k=32, shard_size=1_000_000,
        n_shards=10, total_tokens=10_000_000,
    )
    s = m.to_json()
    m2 = TeacherLogitManifest.from_json(s)
    assert m2 == m
