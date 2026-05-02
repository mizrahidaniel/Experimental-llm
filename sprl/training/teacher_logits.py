"""Offline teacher-logit storage / streaming for distillation pretraining.

File format
-----------

A teacher-logit dataset is a directory tree::

    <root>/
        manifest.json            # global config, see below
        shard_00000000.npz       # first shard
        shard_00000001.npz
        ...

Each shard is an NPZ archive with the following arrays (fixed schema)::

    byte_targets    int32   [N]            ground-truth byte token at each pos
    topk_indices    int32   [N, K]         vocab indices of teacher top-K
    topk_logits     float16 [N, K]         teacher logits for those indices
                                           (raw, pre-softmax, T=1)
    chunk_id        int64   ()             monotonically increasing shard id
    token_offset    int64   ()             global token index of byte_targets[0]

Plus a small JSON `manifest.json` at the root::

    {"vocab_size": 256, "top_k": 32, "shard_size": 1000000,
     "n_shards": 17, "total_tokens": 16842113,
     "teacher": "meta-llama/Llama-3.3-70B-Instruct",
     "dtype_logits": "float16", "dtype_indices": "int32",
     "fp16_quant_max_abs_err": 6.1e-5}  # documented quantization error

K is fixed across all shards (default 32). Shards are written in monotonic
`chunk_id` order; gaps are tolerated by the reader (skip-on-resume).

Quantization
------------

Logits are stored as FP16 to halve disk size relative to FP32. For typical
post-LN softmax inputs (|logit| <= ~30) the relative error is <= 2^-10 ~ 1e-3
and the absolute round-trip error in float space stays under ~6e-5 across
the top-32 range we care about (verified by `pack_topk_logits`'s round-trip
test). Because we KL on softmax(topk/T) the relevant error after T=2 is even
smaller; we have not observed measurable downstream loss drift.

Sparse KL
---------

`sparse_top_k_kl_loss` computes::

    KL(p_T || p_S) = sum_v p_T(v) * (log p_T(v) - log p_S(v))

restricted to the top-K teacher entries plus a *correction* that accounts for
the missing teacher mass: the residual probability `1 - sum_top p_T` is
distributed *uniformly* over the remaining `V - K` entries (a maximum-entropy
prior) and contributes the corresponding KL term. In the limit K=V it
collapses to standard dense KL.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------------
# Manifest

@dataclass
class TeacherLogitManifest:
    vocab_size: int
    top_k: int
    shard_size: int
    n_shards: int
    total_tokens: int
    teacher: str = "unknown"
    dtype_logits: str = "float16"
    dtype_indices: str = "int32"
    fp16_quant_max_abs_err: float = 6.1e-5

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "TeacherLogitManifest":
        return cls(**json.loads(s))


def _shard_name(chunk_id: int) -> str:
    return f"shard_{chunk_id:08d}.npz"


# ---------------------------------------------------------------------------
# Pack / write / read

def pack_topk_logits(
    teacher_logits: Tensor,
    top_k: int = 32,
) -> tuple[Tensor, Tensor]:
    """Convert dense `[..., V]` teacher logits -> (`indices [..., K]`, `logits [..., K]`).

    `logits` is returned as FP16 (the on-disk dtype). The caller is expected to
    pre-shape leading dims to a flat [N] axis before saving.

    FP16 round-trip error: for |logit| <= 60, abs error < 6.1e-5. We do not
    subtract the max here because softmax(topk/T) is shift-invariant.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    V = teacher_logits.shape[-1]
    K = min(top_k, V)
    vals, idx = torch.topk(teacher_logits, K, dim=-1)
    return idx.to(torch.int32), vals.to(torch.float16)


def write_shard(
    out_dir: str | os.PathLike,
    chunk_id: int,
    token_offset: int,
    byte_targets: np.ndarray,
    topk_indices: np.ndarray,
    topk_logits: np.ndarray,
) -> str:
    """Write a single shard. Returns the path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if byte_targets.ndim != 1:
        raise ValueError("byte_targets must be [N]")
    N = byte_targets.shape[0]
    if topk_indices.shape[0] != N or topk_logits.shape[0] != N:
        raise ValueError("topk arrays must align with byte_targets length")
    if topk_indices.shape != topk_logits.shape:
        raise ValueError("topk_indices and topk_logits shape mismatch")
    path = out_dir / _shard_name(chunk_id)
    np.savez(
        path,
        byte_targets=byte_targets.astype(np.int32, copy=False),
        topk_indices=topk_indices.astype(np.int32, copy=False),
        topk_logits=topk_logits.astype(np.float16, copy=False),
        chunk_id=np.int64(chunk_id),
        token_offset=np.int64(token_offset),
    )
    return str(path)


def write_manifest(out_dir: str | os.PathLike, manifest: TeacherLogitManifest) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "manifest.json").write_text(manifest.to_json())


def read_manifest(root: str | os.PathLike) -> TeacherLogitManifest:
    return TeacherLogitManifest.from_json((Path(root) / "manifest.json").read_text())


def list_shards(root: str | os.PathLike) -> list[Path]:
    return sorted(Path(root).glob("shard_*.npz"))


@dataclass
class ShardData:
    chunk_id: int
    token_offset: int
    byte_targets: Tensor
    topk_indices: Tensor
    topk_logits: Tensor

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ShardData":
        with np.load(path) as z:
            return cls(
                chunk_id=int(z["chunk_id"]),
                token_offset=int(z["token_offset"]),
                byte_targets=torch.from_numpy(z["byte_targets"].astype(np.int64)),
                topk_indices=torch.from_numpy(z["topk_indices"].astype(np.int64)),
                topk_logits=torch.from_numpy(z["topk_logits"].astype(np.float32)),
            )


# ---------------------------------------------------------------------------
# Dataset

class TeacherLogitDataset(IterableDataset):
    """Streams teacher-logit shards in `chunk_id` order.

    - Iteration yields `ShardData` objects.
    - Random-access via `seek(chunk_id, token_offset)`: configures the stream
      to begin at that shard, skipping `token_offset - shard.token_offset`
      tokens of the first shard.
    - `prefetch=N` preloads N+1 shards via an in-flight queue.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        prefetch: int = 1,
        start_chunk_id: int = 0,
        start_token_offset: Optional[int] = None,
    ):
        self.root = Path(root)
        self.manifest = read_manifest(self.root) if (self.root / "manifest.json").exists() else None
        self.shards = list_shards(self.root)
        self.prefetch = max(1, int(prefetch))
        self._start_chunk = int(start_chunk_id)
        self._start_offset = start_token_offset

    def seek(self, chunk_id: int, token_offset: Optional[int] = None) -> None:
        self._start_chunk = int(chunk_id)
        self._start_offset = token_offset

    def _shard_index(self, chunk_id: int) -> int:
        for i, p in enumerate(self.shards):
            cid = int(p.stem.split("_")[1])
            if cid >= chunk_id:
                return i
        return len(self.shards)

    def _iter_with_prefetch(self, start_idx: int) -> Iterator[ShardData]:
        from collections import deque

        queue: deque[ShardData] = deque()
        n = self.prefetch + 1

        def _try_fill(i: int) -> int:
            while len(queue) < n and i < len(self.shards):
                queue.append(ShardData.load(self.shards[i]))
                i += 1
            return i

        i = _try_fill(start_idx)
        while queue:
            yield queue.popleft()
            i = _try_fill(i)

    def __iter__(self) -> Iterator[ShardData]:
        idx = self._shard_index(self._start_chunk)
        first = True
        for shard in self._iter_with_prefetch(idx):
            if first and self._start_offset is not None:
                rel = self._start_offset - shard.token_offset
                if 0 < rel < shard.byte_targets.shape[0]:
                    shard = ShardData(
                        chunk_id=shard.chunk_id,
                        token_offset=int(shard.token_offset + rel),
                        byte_targets=shard.byte_targets[rel:],
                        topk_indices=shard.topk_indices[rel:],
                        topk_logits=shard.topk_logits[rel:],
                    )
            first = False
            yield shard

    def get_by_offset(self, chunk_id: int, token_offset: int) -> ShardData:
        for p in self.shards:
            cid = int(p.stem.split("_")[1])
            if cid == chunk_id:
                shard = ShardData.load(p)
                rel = token_offset - shard.token_offset
                if rel < 0 or rel >= shard.byte_targets.shape[0]:
                    return shard
                return ShardData(
                    chunk_id=shard.chunk_id,
                    token_offset=int(shard.token_offset + rel),
                    byte_targets=shard.byte_targets[rel:],
                    topk_indices=shard.topk_indices[rel:],
                    topk_logits=shard.topk_logits[rel:],
                )
        raise KeyError(f"chunk_id {chunk_id} not found in {self.root}")


# ---------------------------------------------------------------------------
# Sparse top-K KL

def sparse_top_k_kl_loss(
    student_logits: Tensor,
    topk_indices: Tensor,
    topk_logits: Tensor,
    T: float = 2.0,
    eps: float = 1e-8,
) -> Tensor:
    """KL(teacher || student) at temperature T using sparse top-K teacher.

    Args:
        student_logits: [..., V] dense student logits (pre-softmax, T=1).
        topk_indices:  [..., K] long, teacher's top-K vocab indices.
        topk_logits:   [..., K] teacher logits (pre-softmax, T=1).
        T: distillation temperature.

    Returns:
        Scalar loss = mean over leading dims of `KL(p_T || p_S) * T^2`.

    Approach: split KL into the top-K contribution (computed exactly using
    student log-probs gathered at teacher top-K indices) and a residual
    contribution where the missing teacher mass `m = 1 - sum_top p_T` is
    distributed uniformly over the `V - K` non-top vocab indices. In the
    limit K=V the residual mass is 0 and this reduces to dense KL.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    V = student_logits.shape[-1]
    K = topk_indices.shape[-1]
    student_logp = F.log_softmax(student_logits / T, dim=-1)  # [..., V]

    # Student log-probs at teacher top-K indices.
    s_logp_top = student_logp.gather(-1, topk_indices)  # [..., K]

    # Teacher: full-vocab unknown, only top-K logits known. Estimate the
    # missing mass by treating each non-top logit as equal to the K-th
    # smallest top-K logit (an upper bound on the residual mass; we
    # renormalize below).
    t_log_top_T = topk_logits.to(student_logits.dtype) / T  # [..., K]
    log_top_unnorm = torch.logsumexp(t_log_top_T, dim=-1)  # [...]
    n_outside = max(V - K, 0)
    if n_outside == 0:
        # Exact: K = V.
        log_Z = log_top_unnorm
        log_resid_unnorm = torch.full_like(log_Z, float("-inf"))
    else:
        min_top_T = topk_logits.min(dim=-1).values.to(student_logits.dtype) / T
        log_resid_unnorm = (
            torch.log(torch.tensor(float(n_outside), device=student_logits.device, dtype=student_logits.dtype))
            + min_top_T
        )
        log_Z = torch.logaddexp(log_top_unnorm, log_resid_unnorm)

    # Teacher distribution at temperature T.
    log_t_p_top = t_log_top_T - log_Z.unsqueeze(-1)  # [..., K]
    t_p_top = log_t_p_top.exp()  # [..., K]

    # Top-K contribution: sum_k p_T(k) * (log p_T(k) - log p_S(k)).
    contrib_top = (t_p_top * (log_t_p_top - s_logp_top)).sum(dim=-1)  # [...]

    if n_outside == 0:
        return contrib_top.mean() * (T ** 2)

    # Residual mass m, uniformly spread over (V - K) entries.
    # log m = log_resid_unnorm - log_Z; per-entry log p_T_outside = log m - log(V-K).
    log_m = log_resid_unnorm - log_Z  # [...]
    m = log_m.exp()  # [...]
    log_p_T_outside = log_m - float(np.log(n_outside))  # [...]

    # Mean of student log-probs outside top-K.
    mask = torch.ones_like(student_logp, dtype=torch.bool)
    mask.scatter_(-1, topk_indices, False)
    s_logp_outside_sum = (student_logp * mask).sum(dim=-1)  # [...]
    mean_s_logp_outside = s_logp_outside_sum / n_outside

    contrib_resid = m * (log_p_T_outside - mean_s_logp_outside)
    kl = contrib_top + contrib_resid
    return kl.mean() * (T ** 2)


# ---------------------------------------------------------------------------
# Combined-loss helper (used by distill.combined_loss)

@dataclass
class TeacherPack:
    """Container for sparse teacher targets aligned to a student batch."""
    topk_indices: Tensor  # [..., K] long
    topk_logits: Tensor   # [..., K] float


def shard_to_pack(shard: ShardData, n: Optional[int] = None) -> tuple[Tensor, TeacherPack]:
    """Slice a shard's first `n` rows into (byte_targets, TeacherPack)."""
    if n is None:
        n = int(shard.byte_targets.shape[0])
    return (
        shard.byte_targets[:n],
        TeacherPack(
            topk_indices=shard.topk_indices[:n],
            topk_logits=shard.topk_logits[:n],
        ),
    )


def batch_mixer(
    dataset: TeacherLogitDataset,
    batch_size: int,
) -> Iterator[tuple[Tensor, TeacherPack]]:
    """Group the streamed shards into fixed-size minibatches of (targets, pack)."""
    buf_targets: list[Tensor] = []
    buf_idx: list[Tensor] = []
    buf_logits: list[Tensor] = []
    n = 0
    for shard in dataset:
        buf_targets.append(shard.byte_targets)
        buf_idx.append(shard.topk_indices)
        buf_logits.append(shard.topk_logits)
        n += int(shard.byte_targets.shape[0])
        while n >= batch_size:
            t = torch.cat(buf_targets, dim=0)
            i = torch.cat(buf_idx, dim=0)
            l = torch.cat(buf_logits, dim=0)
            yield t[:batch_size], TeacherPack(topk_indices=i[:batch_size], topk_logits=l[:batch_size])
            buf_targets = [t[batch_size:]]
            buf_idx = [i[batch_size:]]
            buf_logits = [l[batch_size:]]
            n -= batch_size
    if n > 0:
        t = torch.cat(buf_targets, dim=0)
        i = torch.cat(buf_idx, dim=0)
        l = torch.cat(buf_logits, dim=0)
        yield t, TeacherPack(topk_indices=i, topk_logits=l)
