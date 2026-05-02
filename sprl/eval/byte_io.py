"""Byte <-> patch IO helpers shared by every eval module.

The SPRLv2 forward path expects `[B, S_patches, d_model]` patch embeddings and
emits `byte_logits` of shape `[B, S_patches, max_patch_bytes, vocab=256]`. For
evaluation we need to:

  1. Pack a byte string into fixed-size patches (we don't run BLT entropy
     patching at eval — it depends on the trained byte_lm and is unstable for
     tiny test fixtures). Bytes are right-padded with byte 0 inside the last
     patch; the patch length is tracked so the head's logits at padded slots
     can be ignored.
  2. Embed the per-patch byte windows via the model's `byte_encoder` when the
     patcher is enabled, else via a lightweight learned-free embedding (mean
     of byte embeddings would still require a parameter; we instead require
     `cfg.patcher.enabled=True` for these helpers — the standard config).
  3. Read back per-byte logits from the head and either (a) score a known
     continuation (sum of log-probs) or (b) greedily generate.

These helpers are deterministic given the model state; they assume the model
is in `eval()` mode and run under `torch.no_grad()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from sprl.model import SPRLv2


# ---------------------------------------------------------------------------


def _to_byte_ids(s: str) -> List[int]:
    return list(s.encode("utf-8", errors="replace"))


def _from_byte_ids(ids: List[int]) -> str:
    return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------


def _pack_patches(byte_ids: List[int], max_patch_bytes: int) -> Tuple[Tensor, Tensor]:
    """Greedy fixed-size packing.

    Returns:
      patch_bytes: [N_patches, max_patch_bytes] (256 = pad index — matches
        ByteEncoder padding_idx).
      lengths:    [N_patches] int.
    """
    if not byte_ids:
        # An empty stream still needs one patch so downstream shapes are sane.
        patch_bytes = torch.full((1, max_patch_bytes), 256, dtype=torch.long)
        lengths = torch.tensor([0], dtype=torch.long)
        return patch_bytes, lengths
    n = len(byte_ids)
    n_patches = (n + max_patch_bytes - 1) // max_patch_bytes
    patch_bytes = torch.full((n_patches, max_patch_bytes), 256, dtype=torch.long)
    lengths = torch.zeros(n_patches, dtype=torch.long)
    for p in range(n_patches):
        s = p * max_patch_bytes
        e = min(n, s + max_patch_bytes)
        patch_bytes[p, : e - s] = torch.tensor(byte_ids[s:e], dtype=torch.long)
        lengths[p] = e - s
    return patch_bytes, lengths


@torch.no_grad()
def bytes_to_patch_emb(model: SPRLv2, byte_ids: List[int]) -> Tuple[Tensor, Tensor]:
    """Encode a byte stream into a `[1, S_patches, d_model]` tensor.

    Returns (patch_emb, lengths) where lengths[p] is the true byte count of
    patch p (≤ max_patch_bytes).
    """
    if model.byte_encoder is None:
        raise RuntimeError("byte_io requires cfg.patcher.enabled=True")
    max_pb = model.cfg.patcher.max_patch_bytes
    patch_bytes, lengths = _pack_patches(byte_ids, max_pb)
    device = next(model.parameters()).device
    patch_bytes = patch_bytes.to(device)
    lengths = lengths.to(device)
    # ByteEncoder needs at least length 1 to avoid an all-pad attention crash.
    safe_lengths = lengths.clamp_min(1)
    emb = model.byte_encoder(patch_bytes, safe_lengths)  # [N, d_model]
    return emb.unsqueeze(0), lengths


@torch.no_grad()
def _forward_logits(model: SPRLv2, byte_ids: List[int]) -> Tuple[Tensor, Tensor]:
    """Run the model on a byte stream, return (byte_logits, lengths).

    byte_logits: [S_patches, max_patch_bytes, vocab]
    lengths:     [S_patches]
    """
    patch_emb, lengths = bytes_to_patch_emb(model, byte_ids)
    out = model.forward_from_patches(patch_emb)
    return out.byte_logits[0], lengths  # drop batch dim


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ScoreResult:
    sum_logprob: float       # Σ log p(target byte | context)
    n_bytes: int             # number of scored target bytes
    bits_per_byte: float     # -sum_logprob / (n_bytes * ln 2)


@torch.no_grad()
def score_continuation(
    model: SPRLv2, prompt: str, continuation: str
) -> ScoreResult:
    """Conditional log-probability of `continuation` given `prompt`.

    The byte LM head emits `[S_patches, max_patch_bytes, vocab]` — i.e. given a
    patch's latent it predicts every byte in that patch. To score a byte at
    position t we take the logits from the patch that contains t (its slot
    inside the patch). This treats the model as a within-patch byte
    autoregressor conditioned on the patch latent — consistent with how
    `compute_total_loss` evaluates `byte_targets` against `out.byte_logits`.

    Returns sum log-prob over the continuation bytes only.
    """
    p_ids = _to_byte_ids(prompt)
    c_ids = _to_byte_ids(continuation)
    if not c_ids:
        return ScoreResult(0.0, 0, 0.0)
    full = p_ids + c_ids
    logits, lengths = _forward_logits(model, full)
    log_p = F.log_softmax(logits, dim=-1)  # [S, B_p, V]
    max_pb = model.cfg.patcher.max_patch_bytes

    total = 0.0
    n_scored = 0
    # For every continuation byte i in `full`, look up patch + slot.
    start = len(p_ids)
    end = len(full)
    for i in range(start, end):
        patch_idx = i // max_pb
        slot = i % max_pb
        if patch_idx >= log_p.shape[0]:
            break
        if slot >= int(lengths[patch_idx].item()):
            break
        total += float(log_p[patch_idx, slot, full[i]].item())
        n_scored += 1
    bpb = (-total / (n_scored * 0.6931471805599453)) if n_scored > 0 else 0.0
    return ScoreResult(sum_logprob=total, n_bytes=n_scored, bits_per_byte=bpb)


@torch.no_grad()
def multiple_choice_score(
    model: SPRLv2, prompt: str, choices: List[str], normalize_by_length: bool = True
) -> int:
    """Standard multiple-choice eval: pick the choice with highest log-prob.

    With `normalize_by_length=True` we score by mean log-prob per byte
    (length-normalized), which is the convention for MMLU-style "answer letter
    only" eval but keeps multi-token answers comparable.
    """
    scores: List[float] = []
    for c in choices:
        r = score_continuation(model, prompt, c)
        if r.n_bytes == 0:
            scores.append(float("-inf"))
        elif normalize_by_length:
            scores.append(r.sum_logprob / r.n_bytes)
        else:
            scores.append(r.sum_logprob)
    return int(max(range(len(scores)), key=lambda i: scores[i]))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_bytes(
    model: SPRLv2,
    prompt: str,
    max_new_bytes: int = 64,
    stop: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """Greedy (or low-temp sampled) byte-by-byte generation.

    Re-encodes the full prefix at every step. This is O(n^2) in tokens but
    fine for short generations in unit tests; benchmark code uses
    `decode_speed.py` for timing-sensitive measurement.

    Stops on `stop` substring or after `max_new_bytes`.
    """
    prompt_ids = _to_byte_ids(prompt)
    out_ids: List[int] = []
    max_pb = model.cfg.patcher.max_patch_bytes

    for _ in range(max_new_bytes):
        full = prompt_ids + out_ids
        logits, lengths = _forward_logits(model, full)
        # Next-byte slot is at position len(full) — but we score the *current*
        # byte from its patch latent; for generation we need the next slot.
        nxt = len(full)
        patch_idx = nxt // max_pb
        slot = nxt % max_pb
        # If we'd start a new patch, we still need the latent; re-encode.
        if patch_idx >= logits.shape[0]:
            # Append a placeholder zero byte so encoder runs — its logit is what we want
            full2 = full + [0]
            logits, lengths = _forward_logits(model, full2)
            patch_idx = nxt // max_pb
            slot = nxt % max_pb
        step_logits = logits[patch_idx, slot]
        if temperature <= 0.0:
            nb = int(step_logits.argmax().item())
        else:
            probs = F.softmax(step_logits / temperature, dim=-1)
            nb = int(torch.multinomial(probs, 1).item())
        out_ids.append(nb)
        if stop is not None:
            tail = _from_byte_ids(out_ids)
            if stop in tail:
                # Trim everything from `stop` onward (inclusive of the marker).
                idx = tail.index(stop)
                return tail[:idx]
    return _from_byte_ids(out_ids)
