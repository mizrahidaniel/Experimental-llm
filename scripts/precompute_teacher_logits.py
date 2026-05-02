"""Offline teacher-logit precomputation.

Streams a byte-tokenized dataset through a teacher LM and writes top-K sparse
NPZ shards under <out_dir>. See `sprl.training.teacher_logits` for the format.

Resumable: any shard that already exists on disk is skipped. The shard order
is preserved across resumes.

Mock mode: when `transformers` is missing OR the user passes `--mock-teacher`,
a tiny Markov-chain teacher (per-vocab-pair transition matrix) is used. This
keeps unit tests runnable on CPU without any model downloads.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# Ensure the repo root is on sys.path when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from sprl.training.teacher_logits import (  # noqa: E402
    TeacherLogitManifest,
    pack_topk_logits,
    write_manifest,
    write_shard,
)


# ---------------------------------------------------------------------------
# Teacher loading

def _try_load_hf(model_name: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        return (tok, model)
    except Exception as e:  # pragma: no cover - network-dependent
        print(f"[precompute] HF load failed ({e}); falling back to mock.")
        return None


class MockMarkovTeacher:
    """Tiny vocab-V Markov-chain teacher. Deterministic, no training.

    Logits at position t are a smooth function of byte t-1 only (a hand-rolled
    bigram table). Useful exclusively for exercising the I/O format end-to-end.
    """

    def __init__(self, vocab_size: int = 256, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.V = vocab_size
        self.W = torch.randn(vocab_size, vocab_size, generator=g) * 0.5

    @torch.no_grad()
    def logits_for(self, bytes_in: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([torch.zeros(1, dtype=torch.long), bytes_in[:-1].to(torch.long)])
        return self.W[prev]


# ---------------------------------------------------------------------------
# Byte-stream loader (very small; just enough for the precompute path)

def _load_byte_stream(path: str | os.PathLike) -> torch.Tensor:
    """Reads bytes from a flat file, or a `.npy` of int byte ids."""
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
        return torch.from_numpy(arr.astype(np.int64))
    raw = p.read_bytes()
    return torch.tensor(list(raw), dtype=torch.long)


def _chunked(it: torch.Tensor, n: int) -> Iterable[torch.Tensor]:
    L = it.shape[0]
    for i in range(0, L, n):
        yield it[i : i + n]


# ---------------------------------------------------------------------------
# Main loop

def precompute(
    teacher_name: str,
    byte_dataset: str,
    out_dir: str,
    top_k: int = 32,
    shard_size: int = 1_000_000,
    mock: bool = False,
    vocab_size: int = 256,
) -> int:
    """Returns total shards (newly written + skipped)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    bytes_all = _load_byte_stream(byte_dataset)
    total = int(bytes_all.shape[0])
    n_shards = (total + shard_size - 1) // shard_size

    teacher = None
    if not mock:
        teacher = _try_load_hf(teacher_name)
    if teacher is None:
        print(f"[precompute] using mock Markov teacher (vocab={vocab_size}).")
        teacher = MockMarkovTeacher(vocab_size=vocab_size)
        is_mock = True
    else:
        is_mock = False

    # Wall-clock estimate (rough; informational only).
    per_token_ms = 1e-3 if is_mock else 0.5
    eta_s = total * per_token_ms / 1000.0
    print(
        f"[precompute] tokens={total} shards={n_shards} K={top_k} "
        f"shard_size={shard_size} ETA~{eta_s:.1f}s ({'mock' if is_mock else 'real'})"
    )

    written = 0
    skipped = 0
    for chunk_id in range(n_shards):
        out_path = Path(out_dir) / f"shard_{chunk_id:08d}.npz"
        if out_path.exists():
            skipped += 1
            continue
        start = chunk_id * shard_size
        end = min(start + shard_size, total)
        chunk = bytes_all[start:end]

        if is_mock:
            logits = teacher.logits_for(chunk)
        else:
            tok, model = teacher  # type: ignore
            with torch.no_grad():
                input_ids = chunk.unsqueeze(0)
                out = model(input_ids=input_ids)
                logits = out.logits[0]

        idx, log_fp16 = pack_topk_logits(logits, top_k=top_k)
        write_shard(
            out_dir,
            chunk_id=chunk_id,
            token_offset=start,
            byte_targets=chunk.cpu().numpy().astype(np.int32),
            topk_indices=idx.cpu().numpy(),
            topk_logits=log_fp16.cpu().numpy(),
        )
        written += 1

    manifest = TeacherLogitManifest(
        vocab_size=vocab_size,
        top_k=top_k,
        shard_size=shard_size,
        n_shards=n_shards,
        total_tokens=total,
        teacher=teacher_name if not is_mock else f"mock://markov(V={vocab_size})",
    )
    write_manifest(out_dir, manifest)
    print(f"[precompute] wrote {written} shards, skipped {skipped} existing.")
    return written + skipped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--data", required=True, help="byte stream file or .npy of byte ids")
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--mock-teacher", action="store_true")
    p.add_argument("--vocab-size", type=int, default=256)
    args = p.parse_args()

    t0 = time.time()
    n = precompute(
        teacher_name=args.teacher,
        byte_dataset=args.data,
        out_dir=args.out,
        top_k=args.top_k,
        shard_size=args.shard_size,
        mock=args.mock_teacher,
        vocab_size=args.vocab_size,
    )
    print(f"[precompute] done; {n} shards total in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
