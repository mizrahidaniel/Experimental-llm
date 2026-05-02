"""MathPile byte stream.

HF-or-fallback. Fallback emits arithmetic-expression / proof-line style ASCII
so the patcher sees yet another distribution (digits + math symbols).
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

from sprl.data.fineweb_edu import _try_hf_stream


_OPS = ["+", "-", "*", "/", "=", "<", ">", "<=", ">=", "!="]
_VARS = ["x", "y", "z", "n", "k", "a", "b", "c"]
_FUNCS = ["sin", "cos", "log", "exp", "sqrt", "lim", "sum", "int"]


def _math_like_stream(seed: int) -> Iterator[bytes]:
    rng = random.Random(seed ^ 0x60FE)
    while True:
        out: list[str] = []
        n_lines = rng.randint(2, 6)
        for _ in range(n_lines):
            kind = rng.randint(0, 3)
            if kind == 0:
                a = rng.randint(1, 99)
                b = rng.randint(1, 99)
                op = rng.choice(["+", "-", "*"])
                if op == "+":
                    c = a + b
                elif op == "-":
                    c = a - b
                else:
                    c = a * b
                out.append(f"{a} {op} {b} = {c}\n")
            elif kind == 1:
                v = rng.choice(_VARS)
                out.append(
                    f"{rng.choice(_FUNCS)}({v}) {rng.choice(_OPS)} "
                    f"{rng.randint(0, 9)}\n"
                )
            elif kind == 2:
                out.append(
                    f"For all {rng.choice(_VARS)} in N, "
                    f"{rng.choice(_VARS)}^2 >= 0.\n"
                )
            else:
                out.append(
                    f"Theorem {rng.randint(1, 99)}.{rng.randint(1, 9)}: "
                    f"{rng.choice(_VARS)} = {rng.choice(_VARS)} + "
                    f"{rng.randint(0, 9)}.\n"
                )
        out.append("\n")
        yield "".join(out).encode("utf-8")


def math_pile_stream(
    *,
    seed: int = 0,
    use_hf: bool = True,
    max_examples: Optional[int] = None,
    dataset_name: str = "GAIR/MathPile",
    split: str = "train",
    text_key: str = "text",
) -> Iterator[bytes]:
    if use_hf:
        hf = _try_hf_stream(dataset_name, split, text_key, seed, max_examples)
        if hf is not None:
            yield from hf
            return
    n = 0
    for piece in _math_like_stream(seed):
        if max_examples is not None and n >= max_examples:
            return
        n += 1
        yield piece
