"""Byte-stream utilities and the patcher-aware collator.

A *byte stream* is any iterable yielding `bytes` objects (or anything castable
via `bytes(...)`). The core pipeline is:

    byte stream
      -> chunk_byte_stream(chunk_bytes)            # [T] int tensors
      -> EntropyPatcher.plan(...)                  # PatchPlans
      -> pad to (S_patches, max_patch_bytes)
      -> ByteEncoder(patch_bytes, lengths)         # [B, S, d_model]
      -> SPRLv2.forward_from_patches               # autoregressive byte loss

The collator uses the *frozen* byte LM that ships with `EntropyPatcher` to plan
patches once per chunk (per spec: planning is the expensive step; never call
the patcher per-token).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import torch
from torch import Tensor

from sprl.patcher import ByteEncoder, EntropyPatcher
from sprl.patcher.blt_patcher import PatchPlan

PAD_BYTE = 256  # ByteEncoder uses 256 as pad ID; vocab is 257.
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_byte_files(paths: Sequence[Union[str, Path]]) -> Iterator[bytes]:
    """Yield the raw bytes content of each file (one yield per file)."""
    for p in paths:
        with open(p, "rb") as f:
            yield f.read()


def chunk_byte_stream(
    stream: Iterable[Union[bytes, bytearray, memoryview, str]],
    chunk_bytes: int,
    *,
    drop_last: bool = True,
) -> Iterator[Tensor]:
    """Reshape a byte stream into fixed-size `[chunk_bytes]` int tensors.

    Re-flows across yield boundaries. Strings are UTF-8 encoded.
    """
    buf = bytearray()
    for piece in stream:
        if isinstance(piece, str):
            piece = piece.encode("utf-8", errors="replace")
        buf.extend(piece)
        while len(buf) >= chunk_bytes:
            head = bytes(buf[:chunk_bytes])
            del buf[:chunk_bytes]
            yield torch.tensor(list(head), dtype=torch.long)
    if not drop_last and buf:
        # Pad final partial chunk with PAD_BYTE so downstream shape is stable.
        # PAD_BYTE is 256 (out of native byte range); build via tensor ops.
        out = torch.full((chunk_bytes,), PAD_BYTE, dtype=torch.long)
        out[: len(buf)] = torch.tensor(list(buf), dtype=torch.long)
        yield out


def iter_chunked_bytes(
    stream: Iterable[Union[bytes, bytearray, memoryview, str]],
    chunk_bytes: int,
    *,
    drop_last: bool = True,
) -> Iterator[Tensor]:
    """Alias kept for symmetry with `read_byte_files`."""
    return chunk_byte_stream(stream, chunk_bytes, drop_last=drop_last)


# ---------------------------------------------------------------------------
# Plan -> tensors
# ---------------------------------------------------------------------------


def plans_to_padded_tensors(
    bytes_seq: Tensor,
    plans: List[PatchPlan],
    max_patch_bytes: int,
    max_seq_patches: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert a list of plans + the underlying byte tensor into:

      patch_bytes [B, S, P]    bytes of each patch (PAD_BYTE-padded)
      lengths     [B, S]       true length per patch (>=1 for valid; 0 for pad)
      byte_targets[B, S, P]    same bytes, with IGNORE_INDEX for padded positions
      attn_mask   [B, S]       1 = real patch, 0 = padded patch

    `bytes_seq`: [B, T] integer tensor whose rows correspond 1:1 with `plans`.
    Sequences are truncated to `max_seq_patches` patches.
    """
    assert bytes_seq.dim() == 2, "bytes_seq must be [B, T]"
    B = bytes_seq.shape[0]
    assert len(plans) == B
    S = max_seq_patches
    P = max_patch_bytes

    patch_bytes = torch.full((B, S, P), PAD_BYTE, dtype=torch.long)
    lengths = torch.zeros((B, S), dtype=torch.long)
    byte_targets = torch.full((B, S, P), IGNORE_INDEX, dtype=torch.long)
    attn_mask = torch.zeros((B, S), dtype=torch.long)

    for b, plan in enumerate(plans):
        n = min(len(plan.starts), S)
        for i in range(n):
            s = plan.starts[i]
            e = plan.ends[i]
            ln = min(e - s, P)
            chunk = bytes_seq[b, s : s + ln]
            patch_bytes[b, i, :ln] = chunk
            byte_targets[b, i, :ln] = chunk
            lengths[b, i] = ln
            attn_mask[b, i] = 1

    return patch_bytes, lengths, byte_targets, attn_mask


def derive_mtp_targets(
    byte_targets: Tensor,
    attn_mask: Tensor,
    depth: int,
) -> Tensor:
    """Per-patch MTP targets: the first byte of the next `d` patches.

    Shape `[B, S, depth]`. Positions past the end of the sequence are filled
    with IGNORE_INDEX. We use the first byte of each patch as the MTP target;
    this is a simple, cheap proxy that lines up with `MultiTokenPredictionHead`'s
    expectation of [B, S, depth] integer ids.
    """
    B, S, _ = byte_targets.shape
    out = torch.full((B, S, depth), IGNORE_INDEX, dtype=torch.long)
    for d in range(depth):
        shift = d + 1
        if shift >= S:
            continue
        first = byte_targets[:, shift:, 0]  # [B, S-shift]
        valid = attn_mask[:, shift:].bool()  # [B, S-shift]
        slot = torch.where(valid, first, torch.full_like(first, IGNORE_INDEX))
        out[:, : S - shift, d] = slot
    return out


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass
class CollatorConfig:
    chunk_bytes: int = 512  # raw bytes per "row" before patching
    max_seq_patches: int = 64  # S; truncate plans to this many patches
    max_patch_bytes: int = 16
    batch_size: int = 4
    mtp_depth: int = 2


class PatcherCollator:
    """Pulls bytes from a stream, plans patches, encodes via ByteEncoder.

    Yields dicts shaped exactly as `train_one_step` expects:

        patch_emb     [B, S, d_model]            float (no_grad through encoder)
        byte_targets  [B, S, max_patch_bytes]    long, IGNORE_INDEX on pad
        mtp_targets   [B, S, depth]              long, IGNORE_INDEX on pad
        attention_mask[B, S]                     long, 1 for real patch

    The byte-encoder forward is performed under `torch.no_grad()` because the
    data-loader runs ahead of the trainer; if you want gradients to flow into
    the byte encoder, call it yourself in the training step. (The model in
    this repo treats `patch_emb` as input.)
    """

    def __init__(
        self,
        patcher: EntropyPatcher,
        byte_encoder: ByteEncoder,
        cfg: Optional[CollatorConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.patcher = patcher
        self.byte_encoder = byte_encoder
        self.cfg = cfg or CollatorConfig()
        self.device = device or torch.device("cpu")
        self.patcher.eval()
        self.byte_encoder.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collate(self, byte_chunks: List[Tensor]) -> dict:
        """Take a list of `[chunk_bytes]` int tensors -> a single batch dict."""
        cfg = self.cfg
        bytes_seq = torch.stack(byte_chunks, dim=0).to(self.device)
        plans = self.patcher.plan(bytes_seq)

        patch_bytes, lengths, byte_targets, attn_mask = plans_to_padded_tensors(
            bytes_seq.cpu(),
            plans,
            max_patch_bytes=cfg.max_patch_bytes,
            max_seq_patches=cfg.max_seq_patches,
        )

        B, S, P = patch_bytes.shape
        flat_bytes = patch_bytes.reshape(B * S, P).to(self.device)
        flat_lengths = lengths.reshape(B * S).clamp(min=1).to(self.device)
        # ByteEncoder expects positive lengths even for padded slots; padded
        # patches are masked out via attention_mask downstream.
        emb_flat = self.byte_encoder(flat_bytes, flat_lengths)  # [B*S, d_model]
        patch_emb = emb_flat.reshape(B, S, -1)
        # Zero out padded patches so they cannot leak signal.
        patch_emb = patch_emb * attn_mask.to(patch_emb.dtype).unsqueeze(-1).to(
            patch_emb.device
        )

        mtp_targets = derive_mtp_targets(byte_targets, attn_mask, cfg.mtp_depth)

        return {
            "patch_emb": patch_emb,
            "byte_targets": byte_targets.to(self.device),
            "mtp_targets": mtp_targets.to(self.device),
            "attention_mask": attn_mask.to(self.device),
            "lengths": lengths.to(self.device),
            "patch_bytes": patch_bytes.to(self.device),
        }

    # ------------------------------------------------------------------
    def stream_batches(
        self,
        byte_stream: Iterable[Union[bytes, bytearray, memoryview, str]],
    ) -> Iterator[dict]:
        """Drive the full pipe: byte_stream -> chunked tensors -> batches."""
        cfg = self.cfg
        chunk_iter = chunk_byte_stream(byte_stream, cfg.chunk_bytes, drop_last=True)
        buf: List[Tensor] = []
        for chunk in chunk_iter:
            buf.append(chunk)
            if len(buf) == cfg.batch_size:
                yield self.collate(buf)
                buf = []
        # Drop tail to keep batch size constant (deterministic shape).
