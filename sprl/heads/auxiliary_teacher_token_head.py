"""Auxiliary teacher-token head for distillation across mismatched tokenizers.

When the student and teacher don't share a tokenizer (e.g., student is
byte-level / BLT and teacher is Llama BPE-128K), direct teacher-token KL
between their logits is incoherent. The auxiliary head sits on top of the
student trunk and predicts in the *teacher's* vocabulary; KL is computed
on the aux-head logits.

Gradients flow back into the student trunk, providing distillation signal
without forcing tokenizer alignment. The student's main LM head continues
to train on its own vocab via standard CE.

Per spec §4.1.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AuxiliaryTeacherTokenHead(nn.Module):
    """Linear projection from student trunk hidden → teacher-vocab logits.

    Optional weight-tying with the teacher's input embedding is left to the
    caller (pass `tied_embedding` and the head will use its weight as the
    output projection).
    """

    def __init__(
        self,
        d_model: int,
        teacher_vocab_size: int,
        bias: bool = False,
        tied_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.teacher_vocab_size = teacher_vocab_size
        if tied_embedding is not None:
            assert tied_embedding.embedding_dim == d_model
            assert tied_embedding.num_embeddings >= teacher_vocab_size
            self._tied = tied_embedding
            self.proj: nn.Linear | None = None
        else:
            self._tied = None
            self.proj = nn.Linear(d_model, teacher_vocab_size, bias=bias)

    def forward(self, h: Tensor) -> Tensor:
        """h: [..., d_model] → [..., teacher_vocab_size]."""
        if self._tied is not None:
            w = self._tied.weight[: self.teacher_vocab_size]
            return torch.nn.functional.linear(h, w)
        return self.proj(h)
