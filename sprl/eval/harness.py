"""lm-eval-harness compatibility adapter.

Mirrors the EleutherAI lm-eval-harness `LM` interface enough that a downstream
wrapper is one import line. We intentionally do NOT import lm-eval-harness —
our adapter exposes the same method signatures so users can subclass:

    from lm_eval.api.model import LM
    class SPRLLM(SPRLHarnessAdapter, LM): pass

Methods implemented:
  - loglikelihood(requests) -> List[Tuple[float, bool]]
  - loglikelihood_rolling(requests) -> List[float]
  - generate_until(requests) -> List[str]

Each method accepts either:
  - a list of `Instance`-like objects with an `.args` attribute, OR
  - a list of plain tuples (`(context, continuation)`, `(context, gen_kwargs)`).

This shape matches lm-eval-harness 0.4.x. We don't promise wire-for-wire
parity with future versions; the wrapper is intentionally thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from sprl.eval.byte_io import (
    _forward_logits,
    _to_byte_ids,
    generate_bytes,
    score_continuation,
)
from sprl.model import SPRLv2


def _unwrap_args(req: Any) -> Tuple:
    """Accept either an Instance-like (with .args) or a plain tuple."""
    if hasattr(req, "args"):
        return tuple(req.args)
    if isinstance(req, tuple):
        return req
    raise TypeError(f"unsupported request type: {type(req)}")


@dataclass
class HarnessConfig:
    max_gen_bytes: int = 128
    batch_size: int = 1  # placeholder; loop over requests for now


class SPRLHarnessAdapter:
    """Thin compatibility shim. Subclass alongside `lm_eval.api.model.LM`."""

    def __init__(self, model: SPRLv2, config: Optional[HarnessConfig] = None):
        self.model = model
        self.cfg_h = config or HarnessConfig()
        self.model.eval()

    # ---------- properties expected by harness ---------------------------
    @property
    def device(self) -> str:
        return str(next(self.model.parameters()).device)

    @property
    def eot_token_id(self) -> int:
        return 0  # byte 0 = NUL; not really used at byte level

    @property
    def max_length(self) -> int:
        return self.model.cfg.max_seq_len_patches * self.model.cfg.patcher.max_patch_bytes

    # ---------- core methods --------------------------------------------
    @torch.no_grad()
    def loglikelihood(
        self, requests: Sequence[Any]
    ) -> List[Tuple[float, bool]]:
        """For each (context, continuation) request, return (sum_logprob, is_greedy).

        `is_greedy` is True iff the greedy continuation under the model would
        equal the requested continuation (byte-by-byte). This matches the
        lm-eval-harness convention used for multiple-choice acc-norm.
        """
        results: List[Tuple[float, bool]] = []
        for req in requests:
            ctx, cont = _unwrap_args(req)
            r = score_continuation(self.model, ctx, cont)
            is_greedy = self._is_greedy(ctx, cont)
            results.append((r.sum_logprob, is_greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: Sequence[Any]) -> List[float]:
        """Per-example total log-prob over the entire string (no context)."""
        out: List[float] = []
        for req in requests:
            args = _unwrap_args(req)
            text = args[0]
            r = score_continuation(self.model, "", text)
            out.append(r.sum_logprob)
        return out

    @torch.no_grad()
    def generate_until(self, requests: Sequence[Any]) -> List[str]:
        """For each (context, gen_kwargs) request, generate until a stop string.

        gen_kwargs may include {'until': str|List[str], 'max_gen_toks': int,
        'temperature': float}. We honor 'until' as a single string (using the
        first if a list is given) and 'max_gen_toks' as a max-byte limit.
        """
        out: List[str] = []
        for req in requests:
            args = _unwrap_args(req)
            ctx = args[0]
            kwargs = args[1] if len(args) > 1 else {}
            until = kwargs.get("until")
            if isinstance(until, list):
                stop = until[0] if until else None
            else:
                stop = until
            max_gen = int(kwargs.get("max_gen_toks", self.cfg_h.max_gen_bytes))
            temperature = float(kwargs.get("temperature", 0.0))
            gen = generate_bytes(
                self.model,
                ctx,
                max_new_bytes=max_gen,
                stop=stop,
                temperature=temperature,
            )
            out.append(gen)
        return out

    # ---------- helpers --------------------------------------------------
    @torch.no_grad()
    def _is_greedy(self, ctx: str, cont: str) -> bool:
        """Cheap check: does the model assign max prob to each cont byte?"""
        if not cont:
            return True
        c_ids = _to_byte_ids(cont)
        full_ids = _to_byte_ids(ctx) + c_ids
        logits, lengths = _forward_logits(self.model, full_ids)
        max_pb = self.model.cfg.patcher.max_patch_bytes
        prompt_len = len(_to_byte_ids(ctx))
        for i in range(prompt_len, prompt_len + len(c_ids)):
            patch_idx = i // max_pb
            slot = i % max_pb
            if patch_idx >= logits.shape[0]:
                return False
            if int(logits[patch_idx, slot].argmax().item()) != full_ids[i]:
                return False
        return True

    # ---------- batched convenience -------------------------------------
    def evaluate_request_batch(
        self, requests: Sequence[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Convenience: score a batch of (ctx, cont) pairs and return mean logprob."""
        ll = self.loglikelihood(requests)
        if not ll:
            return {"mean_logprob": 0.0, "n": 0.0}
        total = sum(x[0] for x in ll)
        n = len(ll)
        return {"mean_logprob": total / n, "n_greedy": float(sum(int(x[1]) for x in ll)), "n": float(n)}
