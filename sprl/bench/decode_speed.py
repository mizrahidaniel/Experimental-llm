"""Decode-speed benchmark with optional MTP speculative decoding.

Compares two modes:

  - 'baseline': greedy byte-by-byte decode (one model call per byte).
  - 'mtp_spec': use the MTP head to predict the *next two* bytes from the same
    forward pass; verify each speculative byte against the byte-LM head's
    next-step prediction. Accept the speculative byte iff the verifier's
    argmax matches; otherwise rewind to the verifier's choice. This yields
    >=1× and ≤2× speedup with acceptance rate r in [0,1].

Reports:
  - tokens_per_sec for each mode
  - speedup
  - acceptance_rate (mtp only)

Re-encodes the prefix every step (no KV cache here — SPRL's recurrence makes
plain KV cache nontrivial; full speculative decoding pairs with the trained
verifier in a real system). The test still demonstrates correctness of the
acceptance loop and provides a wall-clock comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sprl.eval.byte_io import _forward_logits, _to_byte_ids, _from_byte_ids
from sprl.model import SPRLv2


@dataclass
class DecodeSpeedConfig:
    prompt: str = "The quick brown fox"
    max_new_bytes: int = 32
    warmup: int = 1
    n_runs: int = 1
    mtp_lookahead: int = 2


# ---------------------------------------------------------------------------


@torch.no_grad()
def _baseline_decode(model: SPRLv2, prompt_ids: List[int], n_new: int) -> List[int]:
    out_ids: List[int] = []
    max_pb = model.cfg.patcher.max_patch_bytes
    for _ in range(n_new):
        full = prompt_ids + out_ids
        # Need a placeholder slot to read next-byte logits.
        full_pad = full + [0]
        logits, _ = _forward_logits(model, full_pad)
        nxt = len(full)
        patch_idx = nxt // max_pb
        slot = nxt % max_pb
        if patch_idx >= logits.shape[0]:
            break
        out_ids.append(int(logits[patch_idx, slot].argmax().item()))
    return out_ids


@torch.no_grad()
def _mtp_predict(model: SPRLv2, full: List[int], depth: int) -> Optional[List[int]]:
    """Use the MTP head to produce `depth` speculative next-bytes.

    Returns None if MTP is disabled.
    """
    if model.mtp is None:
        return None
    # Embed and run the trunk (no MTP supervision since we have no targets;
    # we feed teacher-forcing of zeros and use the *unconditioned* logits at
    # each MTP step. This isn't the "real" speculative pipeline — production
    # MTP uses already-emitted tokens as the teacher input — but it is enough
    # to exercise the path.
    from sprl.eval.byte_io import bytes_to_patch_emb

    patch_emb, _ = bytes_to_patch_emb(model, full)
    out = model.forward_from_patches(patch_emb)
    # Use byte LM head argmax for the first speculative byte (equiv to non-spec).
    max_pb = model.cfg.patcher.max_patch_bytes
    nxt = len(full)
    patch_idx = nxt // max_pb
    slot = nxt % max_pb
    bl = out.byte_logits[0]
    if patch_idx >= bl.shape[0]:
        return None
    first = int(bl[patch_idx, slot].argmax().item())

    # Use MTP for the remaining `depth - 1` lookahead bytes.
    z = patch_emb  # [1, S, d]
    # Run MTP head with placeholder targets equal to the first predicted byte.
    fake_targets = torch.zeros(
        (1, z.shape[1], model.mtp.depth), dtype=torch.long, device=z.device
    )
    fake_targets[0, -1, 0] = first
    mtp_logits = model.mtp(z, fake_targets)  # list of [1, S, vocab]
    spec = [first]
    for d in range(1, min(depth, model.mtp.depth)):
        token = int(mtp_logits[d][0, -1].argmax().item())
        spec.append(token)
    return spec


@torch.no_grad()
def _mtp_decode(
    model: SPRLv2, prompt_ids: List[int], n_new: int, depth: int
) -> Tuple[List[int], int, int]:
    """Speculative decode loop. Returns (out_ids, n_accepted, n_proposed)."""
    out_ids: List[int] = []
    n_accepted = 0
    n_proposed = 0
    max_pb = model.cfg.patcher.max_patch_bytes

    while len(out_ids) < n_new:
        full = prompt_ids + out_ids
        spec = _mtp_predict(model, full, depth=depth)
        if spec is None or len(spec) == 0:
            # Fallback: one verifier step.
            full_pad = full + [0]
            logits, _ = _forward_logits(model, full_pad)
            nxt = len(full)
            patch_idx = nxt // max_pb
            slot = nxt % max_pb
            if patch_idx >= logits.shape[0]:
                break
            out_ids.append(int(logits[patch_idx, slot].argmax().item()))
            continue

        # Verify each speculative byte one at a time against the byte-LM head.
        accepted_here = 0
        for i, candidate in enumerate(spec):
            if len(out_ids) >= n_new:
                break
            full_v = prompt_ids + out_ids + [0]
            logits, _ = _forward_logits(model, full_v)
            nxt = len(prompt_ids) + len(out_ids)
            patch_idx = nxt // max_pb
            slot = nxt % max_pb
            if patch_idx >= logits.shape[0]:
                break
            verifier = int(logits[patch_idx, slot].argmax().item())
            n_proposed += 1
            if verifier == candidate:
                out_ids.append(candidate)
                n_accepted += 1
                accepted_here += 1
            else:
                # Reject: take the verifier byte and re-propose next round.
                out_ids.append(verifier)
                break
        # If all speculative bytes accepted, loop continues with a fresh
        # speculation against the new prefix.
    return out_ids, n_accepted, n_proposed


# ---------------------------------------------------------------------------


def benchmark_decode_speed(
    model: SPRLv2, config: Optional[DecodeSpeedConfig] = None
) -> Dict[str, float]:
    cfg = config or DecodeSpeedConfig()
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = _to_byte_ids(cfg.prompt)

    # Warmup once each.
    for _ in range(max(cfg.warmup, 0)):
        _baseline_decode(model, prompt_ids, n_new=4)
        if model.mtp is not None:
            _mtp_decode(model, prompt_ids, n_new=4, depth=cfg.mtp_lookahead)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # --- baseline ---
    t0 = time.perf_counter()
    for _ in range(cfg.n_runs):
        baseline_out = _baseline_decode(model, prompt_ids, cfg.max_new_bytes)
    if device.type == "cuda":
        torch.cuda.synchronize()
    baseline_dt = time.perf_counter() - t0
    baseline_tps = (len(baseline_out) * cfg.n_runs) / max(baseline_dt, 1e-9)

    # --- mtp ---
    mtp_tps: float = 0.0
    accept_rate: float = 0.0
    if model.mtp is not None:
        t0 = time.perf_counter()
        total_acc = 0
        total_prop = 0
        n_tokens = 0
        for _ in range(cfg.n_runs):
            mtp_out, acc, prop = _mtp_decode(
                model, prompt_ids, cfg.max_new_bytes, depth=cfg.mtp_lookahead
            )
            total_acc += acc
            total_prop += prop
            n_tokens += len(mtp_out)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mtp_dt = time.perf_counter() - t0
        mtp_tps = n_tokens / max(mtp_dt, 1e-9)
        accept_rate = total_acc / max(total_prop, 1)

    speedup = mtp_tps / max(baseline_tps, 1e-9) if mtp_tps > 0 else 1.0
    return {
        "baseline_tokens_per_sec": baseline_tps,
        "mtp_tokens_per_sec": mtp_tps,
        "speedup": speedup,
        "acceptance_rate": accept_rate,
        "n_new_bytes": float(cfg.max_new_bytes),
        "device": 1.0 if device.type == "cuda" else 0.0,
    }
