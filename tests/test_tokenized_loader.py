"""sprl.data.tokenized: shard reader + batch packer + sequence_level loop."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sprl.data.tokenized import (
    TokenizedShardDataset,
    pack_batches,
    tokenized_iterator,
)


def _make_shards(tmp: Path, shards: list[tuple[list[int], list[int]]]):
    """Write a fake shard tree under `tmp`."""
    for i, (token_ids, doc_offsets) in enumerate(shards):
        np.savez_compressed(
            tmp / f"shard_{i:08d}.npz",
            token_ids=np.asarray(token_ids, dtype=np.int32),
            doc_offsets=np.asarray(doc_offsets, dtype=np.int64),
        )
    manifest = {
        "tokenizer_source": "tiny_bpe",
        "vocab_size": 32_000,
        "total_tokens": sum(len(t) for t, _ in shards),
        "total_docs": sum(len(o) - 1 for _, o in shards),
        "n_shards": len(shards),
        "shard_size_target": 1000,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest))


def test_dataset_iterates_shards_in_order(tmp_path):
    _make_shards(tmp_path, [
        ([1, 2, 3], [0, 3]),
        ([4, 5, 6, 7], [0, 4]),
    ])
    ds = TokenizedShardDataset(tmp_path)
    out = list(ds)
    assert [list(t) for t, _ in out] == [[1, 2, 3], [4, 5, 6, 7]]


def test_dataset_vocab_size_from_manifest(tmp_path):
    _make_shards(tmp_path, [([0, 1, 2], [0, 3])])
    ds = TokenizedShardDataset(tmp_path)
    assert ds.vocab_size == 32_000


def test_dataset_missing_shards_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no shard"):
        TokenizedShardDataset(tmp_path)


def test_pack_batches_shape_and_dtype(tmp_path):
    # 24 tokens total → batch=2, seq=4 → 3 batches.
    _make_shards(tmp_path, [(list(range(24)), [0, 24])])
    ds = TokenizedShardDataset(tmp_path)
    batches = list(pack_batches(ds, batch_size=2, seq_len=4))
    assert len(batches) == 3
    for b in batches:
        assert b.shape == (2, 4)
        assert b.dtype == torch.long


def test_pack_batches_packs_across_shards(tmp_path):
    _make_shards(tmp_path, [
        (list(range(10)), [0, 10]),
        (list(range(10, 20)), [0, 10]),
    ])
    ds = TokenizedShardDataset(tmp_path)
    batches = list(pack_batches(ds, batch_size=2, seq_len=5))
    # 20 tokens / (2 * 5) = 2 batches.
    assert len(batches) == 2


def test_pack_batches_doc_respecting_drops_short_tail(tmp_path):
    # Two docs of length 5; seq_len=4 → one full sequence per doc, tail dropped.
    _make_shards(tmp_path, [(list(range(10)), [0, 5, 10])])
    ds = TokenizedShardDataset(tmp_path)
    batches = list(pack_batches(ds, batch_size=2, seq_len=4, pack_across_docs=False))
    # 2 doc-slices of length 4 = 1 batch of size 2.
    assert len(batches) == 1
    assert batches[0].shape == (2, 4)


def test_tokenized_iterator_emits_mtp_targets(tmp_path):
    _make_shards(tmp_path, [(list(range(32)), [0, 32])])
    it = tokenized_iterator(tmp_path, batch_size=2, seq_len=8)
    batch = next(iter(it))
    assert "token_ids" in batch and "mtp_targets" in batch
    tok = batch["token_ids"]
    mtp = batch["mtp_targets"]
    assert tok.shape == (2, 8)
    assert mtp.shape == (2, 8, 2)
    # mtp_targets[..., 0] is the next token (shift-by-1); last position is -100.
    assert torch.equal(mtp[:, : 7, 0], tok[:, 1:])
    assert (mtp[:, 7, 0] == -100).all()
    # mtp_targets[..., 1] is shift-by-2; last 2 positions -100.
    assert torch.equal(mtp[:, : 6, 1], tok[:, 2:])
    assert (mtp[:, 6:, 1] == -100).all()


def test_shuffle_is_deterministic(tmp_path):
    _make_shards(tmp_path, [
        ([i], [0, 1]) for i in range(5)
    ])
    a = list(TokenizedShardDataset(tmp_path, shuffle_shards=True, seed=7))
    b = list(TokenizedShardDataset(tmp_path, shuffle_shards=True, seed=7))
    assert [list(x) for x, _ in a] == [list(x) for x, _ in b]
