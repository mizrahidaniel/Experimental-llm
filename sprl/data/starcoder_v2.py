"""StarCoder-v2 byte stream.

HF-or-fallback. Fallback emits structured Python-like code so the BLT entropy
patcher sees a different distribution than English (more punctuation, more
line-breaks).
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

from sprl.data.fineweb_edu import _try_hf_stream


_KEYWORDS = [
    "def", "return", "for", "while", "if", "elif", "else", "import",
    "from", "class", "yield", "break", "continue", "pass", "with", "as",
    "raise", "try", "except", "finally", "lambda", "in", "not", "and", "or",
]
_FUNCS = ["foo", "bar", "process", "load", "save", "run", "step", "encode", "decode"]
_ARGS = ["x", "y", "z", "n", "i", "j", "k", "data", "buf", "cfg"]
_OPS = [" + ", " - ", " * ", " / ", " == ", " != ", " < ", " > ", " <= ", " >= "]


def _code_like_stream(seed: int) -> Iterator[bytes]:
    rng = random.Random(seed ^ 0x5C0DE)
    while True:
        out: list[str] = []
        n_funcs = rng.randint(1, 3)
        for _ in range(n_funcs):
            fname = rng.choice(_FUNCS) + str(rng.randint(0, 9))
            n_args = rng.randint(1, 3)
            args = [rng.choice(_ARGS) for _ in range(n_args)]
            out.append(f"def {fname}({', '.join(args)}):\n")
            for _ in range(rng.randint(2, 6)):
                indent = "    " * rng.randint(1, 2)
                stmt_kind = rng.randint(0, 3)
                if stmt_kind == 0:
                    a = rng.choice(_ARGS)
                    b = rng.choice(_ARGS)
                    out.append(f"{indent}{a}{rng.choice(_OPS)}{b}\n")
                elif stmt_kind == 1:
                    out.append(f"{indent}{rng.choice(_KEYWORDS)} {rng.choice(_ARGS)}\n")
                elif stmt_kind == 2:
                    out.append(
                        f"{indent}{rng.choice(_ARGS)} = {rng.choice(_FUNCS)}("
                        f"{rng.choice(_ARGS)})\n"
                    )
                else:
                    out.append(f"{indent}# {rng.choice(_KEYWORDS)} {rng.choice(_FUNCS)}\n")
            out.append(f"    return {rng.choice(_ARGS)}\n\n")
        yield "".join(out).encode("utf-8")


def starcoder_v2_stream(
    *,
    seed: int = 0,
    use_hf: bool = True,
    max_examples: Optional[int] = None,
    dataset_name: str = "bigcode/the-stack-v2",
    split: str = "train",
    text_key: str = "content",
) -> Iterator[bytes]:
    if use_hf:
        hf = _try_hf_stream(dataset_name, split, text_key, seed, max_examples)
        if hf is not None:
            yield from hf
            return
    n = 0
    for piece in _code_like_stream(seed):
        if max_examples is not None and n >= max_examples:
            return
        n += 1
        yield piece
