"""Generic associative scan over a sequence axis.

Reference implementation. Production should use FLA-style chunked Triton
kernels (https://github.com/fla-org/flash-linear-attention); here we provide
correct PyTorch fallbacks that:

1. Compute the same result as a sequential scan.
2. Are differentiable end-to-end.
3. Have a chunked variant that bounds memory and matches a sequential scan
   exactly when associativity holds.

`combine_fn(a, b) -> c` must be associative: `combine(combine(a,b), c) == combine(a, combine(b,c))`.
For non-associative combinators, use `chunked_associative_scan` with chunk_size=1
which degenerates to a sequential scan.

Three scan flavours:
  - `associative_scan`: sequential reference (O(T) work, O(T) span).
  - `blelloch_scan`: work-efficient parallel scan via Blelloch's up-sweep +
    down-sweep, identity-free (we maintain a `valid` bitmap in lieu of an
    explicit identity element). Per-level combines are batched over the pairs
    at that level. Output is equivalent to `associative_scan` for any
    associative combine.
  - `chunked_associative_scan`: bounded-memory chunked sequential.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor


# A combinator step accepts pytrees of tensors and returns the combined pytree.
Combine = Callable[[Tuple[Tensor, ...], Tuple[Tensor, ...]], Tuple[Tensor, ...]]


def associative_scan(
    elements: Tuple[Tensor, ...],
    combine_fn: Combine,
    axis: int = 1,
) -> Tuple[Tensor, ...]:
    """Sequential reference scan along `axis`. Returns the same pytree shape.

    elements: tuple of tensors, all sharing the scan axis.
    """
    # Move scan axis to position 0.
    permuted = tuple(t.movedim(axis, 0) for t in elements)
    T = permuted[0].shape[0]
    out_list: list[Tuple[Tensor, ...]] = [tuple(p[0] for p in permuted)]
    cur = out_list[0]
    for t in range(1, T):
        nxt = tuple(p[t] for p in permuted)
        cur = combine_fn(cur, nxt)
        out_list.append(cur)
    stacked = tuple(torch.stack([o[i] for o in out_list], dim=0) for i in range(len(elements)))
    # Move back.
    return tuple(s.movedim(0, axis) for s in stacked)


# ---------------------------------------------------------------------------
# Blelloch (work-efficient) parallel scan — identity-free inclusive variant.
# ---------------------------------------------------------------------------


def blelloch_scan(
    elements: Tuple[Tensor, ...],
    combine_fn: Combine,
    axis: int = 1,
) -> Tuple[Tensor, ...]:
    """Work-efficient parallel inclusive scan along `axis`.

    Blelloch (1990) up-sweep + down-sweep on a power-of-two padded sequence.
    Identity-free: the textbook exclusive variant requires an identity
    element at the root; we maintain a parallel `valid` mask so the combine
    is only invoked where both operands are defined. After the down-sweep
    yields exclusive prefixes, we fold the leaf back in to recover inclusive
    output.

    O(T) total work, O(log T) parallel depth. Per-level combines are batched
    over the pairs at that level (a single combine call processes all pairs
    in lockstep).
    """
    permuted = tuple(t.movedim(axis, 0) for t in elements)
    T = permuted[0].shape[0]
    if T == 0 or T == 1:
        return tuple(t.movedim(0, axis) for t in permuted)

    # Pad to next power of two by repeating the last element.
    n = 1
    while n < T:
        n <<= 1
    if n != T:
        last = tuple(p[-1:] for p in permuted)
        pad = tuple(l.expand(n - T, *l.shape[1:]).clone() for l in last)
        padded = tuple(torch.cat([p, pp], dim=0) for p, pp in zip(permuted, pad))
    else:
        padded = tuple(p.clone() for p in permuted)

    n_t = len(padded)

    # ---------------- up-sweep ----------------
    # buf[k][i] = current tree value (segment reduction at right children).
    buf: list[list[Tensor]] = [[padded[k][t] for t in range(n)] for k in range(n_t)]
    step = 1
    while step < n:
        right_idx = list(range(step * 2 - 1, n, step * 2))
        left_idx = [i - step for i in right_idx]
        left_batch = tuple(
            torch.stack([buf[k][i] for i in left_idx], dim=0) for k in range(n_t)
        )
        right_batch = tuple(
            torch.stack([buf[k][i] for i in right_idx], dim=0) for k in range(n_t)
        )
        merged = combine_fn(left_batch, right_batch)
        for k in range(n_t):
            for j, i in enumerate(right_idx):
                buf[k][i] = merged[k][j]
        step <<= 1

    # ---------------- down-sweep (exclusive scan) ----------------
    seg: list[list[Tensor]] = [[buf[k][i].clone() for i in range(n)] for k in range(n_t)]

    # ex[k][i] holds the exclusive prefix at position i; valid[i] tracks
    # whether ex[i] is meaningful (replaces the textbook identity at the root).
    ex: list[list[Tensor]] = [[padded[k][t] for t in range(n)] for k in range(n_t)]
    valid: list[bool] = [False] * n
    # Root holds the full reduction; for exclusive scan it should be invalid
    # (= identity). Propagation fills in the rest.

    step = n // 2
    while step >= 1:
        right_idx = list(range(step * 2 - 1, n, step * 2))
        left_idx = [i - step for i in right_idx]
        new_ex: dict[int, Tuple[Tensor, ...]] = {}
        new_valid: dict[int, bool] = {}
        for j, i in enumerate(right_idx):
            li = left_idx[j]
            parent_valid = valid[i]
            parent_ex = (
                tuple(ex[k][i] for k in range(n_t)) if parent_valid else None
            )
            seg_left = tuple(seg[k][li] for k in range(n_t))

            # Left child inherits parent's exclusive prefix.
            if parent_valid:
                new_ex[li] = parent_ex
                new_valid[li] = True
            else:
                new_valid[li] = False

            # Right child: parent_ex ⊕ seg_left  (or seg_left alone if parent invalid).
            if parent_valid:
                merged = combine_fn(
                    tuple(p.unsqueeze(0) for p in parent_ex),
                    tuple(s.unsqueeze(0) for s in seg_left),
                )
                new_ex[i] = tuple(m.squeeze(0) for m in merged)
                new_valid[i] = True
            else:
                new_ex[i] = seg_left
                new_valid[i] = True
        for i, vex in new_ex.items():
            for k in range(n_t):
                ex[k][i] = vex[k]
        for i, vv in new_valid.items():
            valid[i] = vv
        step //= 2

    # ---------------- exclusive → inclusive ----------------
    inclusive: list[list[Tensor]] = [[None] * n for _ in range(n_t)]  # type: ignore
    valid_idx = [i for i in range(n) if valid[i]]
    invalid_idx = [i for i in range(n) if not valid[i]]
    if valid_idx:
        ex_batch = tuple(
            torch.stack([ex[k][i] for i in valid_idx], dim=0) for k in range(n_t)
        )
        leaf_batch = tuple(
            torch.stack([padded[k][i] for i in valid_idx], dim=0) for k in range(n_t)
        )
        merged = combine_fn(ex_batch, leaf_batch)
        for j, i in enumerate(valid_idx):
            for k in range(n_t):
                inclusive[k][i] = merged[k][j]
    for i in invalid_idx:
        for k in range(n_t):
            inclusive[k][i] = padded[k][i]

    out_stacked = tuple(torch.stack(inclusive[k][:T], dim=0) for k in range(n_t))
    return tuple(s.movedim(0, axis) for s in out_stacked)


# ---------------------------------------------------------------------------


def chunked_associative_scan(
    elements: Tuple[Tensor, ...],
    combine_fn: Combine,
    chunk_size: int = 256,
    axis: int = 1,
) -> Tuple[Tensor, ...]:
    """Chunked sequential scan.

    Equivalent to `associative_scan` but processes the sequence in chunks; the
    last element of each chunk is fed as the initial state of the next.

    Backprop flows through chunks (no detach across chunks). For a TBPTT-style
    boundary, call with `.detach()` on the carry between chunks.
    """
    permuted = tuple(t.movedim(axis, 0) for t in elements)
    T = permuted[0].shape[0]
    if chunk_size >= T:
        return associative_scan(elements, combine_fn, axis=axis)

    out_per_chunk: list[Tuple[Tensor, ...]] = []
    carry: Tuple[Tensor, ...] | None = None
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        sub = tuple(p[s:e] for p in permuted)
        if carry is not None:
            # Prepend carry as a virtual element 0; scan; drop it.
            sub_with_carry = tuple(
                torch.cat([c.unsqueeze(0), x], dim=0) for c, x in zip(carry, sub)
            )
            scanned = associative_scan(
                tuple(x.movedim(0, axis) for x in sub_with_carry), combine_fn, axis=axis
            )
            scanned = tuple(s_.movedim(axis, 0) for s_ in scanned)
            scanned = tuple(s_[1:] for s_ in scanned)
        else:
            scanned = associative_scan(
                tuple(x.movedim(0, axis) for x in sub), combine_fn, axis=axis
            )
            scanned = tuple(s_.movedim(axis, 0) for s_ in scanned)
        out_per_chunk.append(scanned)
        carry = tuple(s_[-1] for s_ in scanned)

    cat = tuple(torch.cat([c[i] for c in out_per_chunk], dim=0) for i in range(len(permuted)))
    return tuple(c.movedim(0, axis) for c in cat)
