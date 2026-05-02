"""Tokenized-shard loader for the BPE training path.

Reads the NPZ shard tree produced by `scripts/tokenize_corpus.py` and yields
training batches of `[B, S]` int32 token IDs. Document boundaries from the
tokenizer pass are honored: sequences are packed without crossing a doc
boundary unless `pack_across_docs=True`.

Pairs with `SPRLv2.forward_from_token_ids(token_ids)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch


@dataclass
class TokenizedManifest:
    tokenizer_source: str
    vocab_size: int
    total_tokens: int
    total_docs: int
    n_shards: int
    shard_size_target: int


def read_manifest(root: Path) -> TokenizedManifest:
    with (root / "manifest.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return TokenizedManifest(**{
        k: raw.get(k) for k in TokenizedManifest.__dataclass_fields__
    })


class TokenizedShardDataset:
    """Streams tokenized NPZ shards in order. Reentrant; deterministic with a seed.

    Yields `(token_ids, doc_offsets)` per shard. Higher-level code packs these
    into `[B, S]` batches via `pack_batches`.
    """

    def __init__(self, root: str | Path, shuffle_shards: bool = False, seed: int = 0):
        self.root = Path(root)
        self.shards = sorted(self.root.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no shard_*.npz files under {self.root}")
        if shuffle_shards:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(self.shards)).tolist()
            self.shards = [self.shards[i] for i in order]
        self._manifest: Optional[TokenizedManifest] = None
        if (self.root / "manifest.json").exists():
            self._manifest = read_manifest(self.root)

    @property
    def vocab_size(self) -> int:
        if self._manifest is None:
            raise RuntimeError("manifest.json missing; cannot infer vocab_size")
        return self._manifest.vocab_size

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for shard in self.shards:
            with np.load(shard) as data:
                yield data["token_ids"].copy(), data["doc_offsets"].copy()


def pack_batches(
    dataset: TokenizedShardDataset,
    batch_size: int,
    seq_len: int,
    pack_across_docs: bool = True,
    drop_last: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield `[B, S]` int64 tensors from a stream of shards.

    pack_across_docs=True (default): treat each shard as one long stream;
        cheap, ignores doc boundaries.
    pack_across_docs=False: chunk each document independently; tail tokens
        below `seq_len` are dropped.
    """
    if pack_across_docs:
        buf: List[np.ndarray] = []
        total = 0
        for token_ids, _ in dataset:
            buf.append(token_ids)
            total += token_ids.size
            need = batch_size * seq_len
            while total >= need:
                stream = np.concatenate(buf)
                head = stream[:need]
                tail = stream[need:]
                buf = [tail] if tail.size else []
                total = tail.size
                yield torch.from_numpy(head.reshape(batch_size, seq_len)).long()
        if not drop_last and total >= seq_len:
            stream = np.concatenate(buf)
            n_full = total // seq_len
            head = stream[: n_full * seq_len].reshape(n_full, seq_len)
            # Pad to batch_size with zeros so the consumer sees a full batch.
            pad = batch_size - n_full % batch_size if n_full % batch_size else 0
            if pad:
                head = np.concatenate([head, np.zeros((pad, seq_len), dtype=head.dtype)])
            for i in range(0, head.shape[0], batch_size):
                yield torch.from_numpy(head[i : i + batch_size]).long()
        return

    # Doc-respecting path.
    buf_seqs: List[np.ndarray] = []
    for token_ids, doc_offsets in dataset:
        for i in range(len(doc_offsets) - 1):
            doc = token_ids[doc_offsets[i] : doc_offsets[i + 1]]
            for s in range(0, doc.size - seq_len + 1, seq_len):
                buf_seqs.append(doc[s : s + seq_len])
                if len(buf_seqs) == batch_size:
                    yield torch.from_numpy(np.stack(buf_seqs)).long()
                    buf_seqs = []
    if not drop_last and buf_seqs:
        # Pad tail with zeros to batch_size.
        while len(buf_seqs) < batch_size:
            buf_seqs.append(np.zeros(seq_len, dtype=buf_seqs[0].dtype))
        yield torch.from_numpy(np.stack(buf_seqs)).long()


def tokenized_iterator(
    root: str | Path,
    batch_size: int,
    seq_len: int,
    shuffle_shards: bool = False,
    seed: int = 0,
) -> Iterator[dict]:
    """High-level iterator yielding {`token_ids`, `mtp_targets`} dicts that
    drop straight into `train_one_step` for the BPE path.

    `mtp_targets` predicts the next-2 tokens at each position; tail positions
    are filled with -100 (ignore_index) so they don't contribute to MTP loss.
    """
    ds = TokenizedShardDataset(root, shuffle_shards=shuffle_shards, seed=seed)
    for token_ids in pack_batches(ds, batch_size, seq_len):
        # Build MTP targets: shift by 1 and 2 positions; pad tail with -100.
        B, S = token_ids.shape
        targets_t1 = torch.full_like(token_ids, -100)
        targets_t2 = torch.full_like(token_ids, -100)
        targets_t1[:, : S - 1] = token_ids[:, 1:]
        targets_t2[:, : S - 2] = token_ids[:, 2:]
        mtp = torch.stack([targets_t1, targets_t2], dim=-1)  # [B, S, 2]
        yield {"token_ids": token_ids, "mtp_targets": mtp}
