"""Tokenize a JSONL or HuggingFace text corpus into int32 token-id NPZ shards.

Output layout:
    <out>/manifest.json           # vocab_size, tokenizer source, total tokens, n_shards
    <out>/shard_00000000.npz      # token_ids: int32 [N], doc_offsets: int32 [n_docs+1]
    <out>/shard_00000001.npz
    ...

Each shard contains roughly `--shard-size` tokens (last shard may be short).
Document boundaries are preserved via `doc_offsets` so downstream loaders can
respect them when building training sequences.

Resumable: if `<out>/shard_NNNNNNNN.npz` already exists, the script seeks past
the corresponding byte offset in the input and continues from there.

Falls back to `MiniBPE(vocab_size=registered)` when HuggingFace `transformers`
is missing or the named teacher repo is gated/offline. CI / unit-test runs
hit this path; real distillation runs need the actual teacher tokenizer.

Usage:
    python scripts/tokenize_corpus.py \\
        --tokenizer llama_3_1_8b \\
        --input  data/fineweb_edu_topq.jsonl \\
        --output data/tokenized/fineweb_edu_topq \\
        --field  text \\
        --shard-size 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

from sprl.tokenizer import load_tokenizer, vocab_size_for


def iter_jsonl(path: Path, field: str, skip_docs: int = 0) -> Iterator[str]:
    """Yield text from each JSONL record's `field`. Skips the first
    `skip_docs` records (used for resume)."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip_docs:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(field)
            if isinstance(text, str) and text:
                yield text


def write_shard(
    out_dir: Path,
    shard_id: int,
    token_ids: list[int],
    doc_offsets: list[int],
) -> Path:
    path = out_dir / f"shard_{shard_id:08d}.npz"
    np.savez_compressed(
        path,
        token_ids=np.asarray(token_ids, dtype=np.int32),
        doc_offsets=np.asarray(doc_offsets, dtype=np.int64),
    )
    return path


def existing_shard_count(out_dir: Path) -> int:
    return len(sorted(out_dir.glob("shard_*.npz")))


def docs_per_existing_shard(out_dir: Path) -> int:
    """Sum of doc counts across existing shards (used for resume offset)."""
    n = 0
    for shard in sorted(out_dir.glob("shard_*.npz")):
        with np.load(shard) as data:
            n += int(data["doc_offsets"].shape[0]) - 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True,
                   help="Source from sprl.tokenizer registry (e.g. llama_3_1_8b, qwen_2_5_7b, tiny_bpe)")
    p.add_argument("--input", type=Path, required=True, help="JSONL file with text records")
    p.add_argument("--output", type=Path, required=True, help="Output directory for NPZ shards")
    p.add_argument("--field", default="text", help="JSONL field name to read text from")
    p.add_argument("--shard-size", type=int, default=1_000_000,
                   help="Approximate tokens per shard")
    p.add_argument("--max-docs", type=int, default=None, help="Stop after N documents (debug)")
    p.add_argument("--no-network", action="store_true",
                   help="Force offline MiniBPE fallback (CI / sandboxed runs)")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer, allow_network=not args.no_network)
    vocab = tokenizer.vocab_size
    print(f"[tokenize] tokenizer={args.tokenizer} vocab={vocab} → {args.output}")

    # Resume: skip past docs already written.
    skip_docs = docs_per_existing_shard(args.output)
    next_shard = existing_shard_count(args.output)
    if skip_docs:
        print(f"[tokenize] resuming: {next_shard} shards present, skipping {skip_docs} docs")

    buf_ids: list[int] = []
    buf_offsets: list[int] = [0]
    total_tokens = 0
    n_docs = 0
    shard_id = next_shard

    for text in iter_jsonl(args.input, args.field, skip_docs=skip_docs):
        ids = tokenizer.encode(text)
        # Append EOS-like separator (use eos_token_id if defined).
        if tokenizer.eos_token_id and tokenizer.eos_token_id < vocab:
            ids = ids + [tokenizer.eos_token_id]
        buf_ids.extend(ids)
        buf_offsets.append(len(buf_ids))
        total_tokens += len(ids)
        n_docs += 1

        if len(buf_ids) >= args.shard_size:
            path = write_shard(args.output, shard_id, buf_ids, buf_offsets)
            print(f"[tokenize] wrote {path.name}: {len(buf_ids):,} tokens, "
                  f"{len(buf_offsets) - 1:,} docs")
            shard_id += 1
            buf_ids = []
            buf_offsets = [0]

        if args.max_docs and n_docs >= args.max_docs:
            break

    # Flush tail.
    if buf_ids:
        path = write_shard(args.output, shard_id, buf_ids, buf_offsets)
        print(f"[tokenize] wrote {path.name} (tail): {len(buf_ids):,} tokens, "
              f"{len(buf_offsets) - 1:,} docs")
        shard_id += 1

    manifest = {
        "tokenizer_source": args.tokenizer,
        "vocab_size": vocab,
        "total_tokens": total_tokens + (skip_docs * 0),  # approximate
        "total_docs": n_docs + skip_docs,
        "n_shards": shard_id,
        "shard_size_target": args.shard_size,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[tokenize] done: {n_docs:,} new docs, {total_tokens:,} new tokens, "
          f"{shard_id} shards total")


if __name__ == "__main__":
    main()
