"""Byte-level perplexity / bits-per-byte over a held-out byte stream.

Mirrors the within-patch byte cross-entropy used by `compute_total_loss`. We
chunk the stream into windows of `chunk_patches` patches (× max_patch_bytes)
and average per-byte NLL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from sprl.eval.byte_io import _pack_patches, _to_byte_ids
from sprl.model import SPRLv2


@dataclass
class PerplexityConfig:
    chunk_patches: int = 16          # patches per forward
    max_chunks: Optional[int] = None  # cap evaluation, None = full stream
    pad_byte: int = 0


def _byte_stream_to_ids(stream: Union[str, bytes, Sequence[int]]) -> List[int]:
    if isinstance(stream, str):
        return _to_byte_ids(stream)
    if isinstance(stream, (bytes, bytearray)):
        return list(stream)
    return [int(x) & 0xFF for x in stream]


@torch.no_grad()
def evaluate_perplexity(
    model: SPRLv2,
    stream: Union[str, bytes, Sequence[int], Iterable[Union[str, bytes]]],
    config: Optional[PerplexityConfig] = None,
) -> Dict[str, float]:
    """Compute byte-level perplexity over a held-out stream.

    The stream may be a string, bytes, a list of ints, or an iterable of
    strings/byte-chunks (concatenated). Variable-length packing is handled
    by truncating the last patch to its real byte count and ignoring padded
    slot logits.

    Returns: {nll, ppl, bpb, n_bytes, n_patches}
    """
    cfg = config or PerplexityConfig()
    model.eval()

    if not isinstance(stream, (str, bytes, bytearray)) and hasattr(stream, "__iter__") and not isinstance(stream, list):
        # Iterable of chunks.
        all_ids: List[int] = []
        for chunk in stream:
            all_ids.extend(_byte_stream_to_ids(chunk))
        ids = all_ids
    else:
        ids = _byte_stream_to_ids(stream)

    if not ids:
        return {"nll": 0.0, "ppl": 1.0, "bpb": 0.0, "n_bytes": 0.0, "n_patches": 0.0}

    max_pb = model.cfg.patcher.max_patch_bytes
    chunk_bytes = cfg.chunk_patches * max_pb

    total_nll = 0.0
    total_n = 0
    total_patches = 0
    n_chunks = 0
    device = next(model.parameters()).device

    for s in range(0, len(ids), chunk_bytes):
        if cfg.max_chunks is not None and n_chunks >= cfg.max_chunks:
            break
        chunk_ids = ids[s : s + chunk_bytes]
        patch_bytes, lengths = _pack_patches(chunk_ids, max_pb)
        patch_bytes = patch_bytes.to(device)
        lengths = lengths.to(device)
        emb = model.byte_encoder(patch_bytes, lengths.clamp_min(1))
        out = model.forward_from_patches(emb.unsqueeze(0))
        logits = out.byte_logits[0]  # [S, B_p, V]
        log_p = F.log_softmax(logits, dim=-1)
        S, Bp, V = logits.shape

        # Build per-slot mask of valid bytes.
        slot_idx = torch.arange(Bp, device=device).unsqueeze(0)
        valid = slot_idx < lengths.unsqueeze(1)  # [S, B_p]
        targets = patch_bytes.clone()
        # Replace pad with 0 so gather doesn't oob; mask will zero its loss.
        targets = torch.where(valid, targets, torch.zeros_like(targets))
        gathered = log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [S, B_p]
        nll_per_byte = -gathered
        masked = nll_per_byte * valid.float()
        total_nll += float(masked.sum().item())
        total_n += int(valid.sum().item())
        total_patches += S
        n_chunks += 1

    if total_n == 0:
        return {"nll": 0.0, "ppl": 1.0, "bpb": 0.0, "n_bytes": 0.0, "n_patches": 0.0}

    mean_nll = total_nll / total_n
    bpb = mean_nll / math.log(2.0)
    ppl = math.exp(min(mean_nll, 50.0))  # clamp to avoid overflow on garbage
    return {
        "nll": mean_nll,
        "ppl": ppl,
        "bpb": bpb,
        "n_bytes": float(total_n),
        "n_patches": float(total_patches),
    }
