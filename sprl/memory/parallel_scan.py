"""Generic associative scan over a sequence axis.

Reference implementation. Production should use FLA-style chunked Triton
kernels (https://github.com/fla-org/flash-linear-attention); here we provide a
correct PyTorch fallback that:

1. Computes the same result as a sequential scan.
2. Is differentiable end-to-end.
3. Has a chunked variant that bounds memory and matches a sequential scan
   exactly when associativity holds.

`combine_fn(a, b) -> c` must be associative: `combine(combine(a,b), c) == combine(a, combine(b,c))`.
For non-associative combinators, use `chunked_associative_scan` with chunk_size=1
which degenerates to a sequential scan.
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
