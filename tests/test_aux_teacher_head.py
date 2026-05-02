"""Auxiliary teacher-token head + KL direction utilities."""

import torch
import torch.nn.functional as F

from sprl.heads import AuxiliaryTeacherTokenHead
from sprl.training.distill import distillation_kl_loss, select_teacher_for_phase


def test_aux_head_shape():
    head = AuxiliaryTeacherTokenHead(d_model=64, teacher_vocab_size=32_000)
    h = torch.randn(4, 16, 64)
    logits = head(h)
    assert logits.shape == (4, 16, 32_000)


def test_aux_head_grad_flows_to_trunk():
    head = AuxiliaryTeacherTokenHead(d_model=32, teacher_vocab_size=128)
    h = torch.randn(2, 8, 32, requires_grad=True)
    logits = head(h)
    loss = logits.float().mean()
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_aux_head_tied_embedding():
    """Tied-embedding mode reuses the input embedding's weight matrix."""
    embed = torch.nn.Embedding(num_embeddings=5000, embedding_dim=32)
    head = AuxiliaryTeacherTokenHead(
        d_model=32, teacher_vocab_size=2000, tied_embedding=embed
    )
    h = torch.randn(1, 4, 32)
    logits = head(h)
    assert logits.shape == (1, 4, 2000)
    # Manually compute the expected logits: h @ W[:V_aux].T
    expected = h @ embed.weight[:2000].t()
    assert torch.allclose(logits, expected, atol=1e-5)


def test_kl_direction_forward_vs_reverse_disagree():
    s = torch.randn(4, 8, 16)
    t = torch.randn(4, 8, 16) + 1.0  # shifted, so distributions differ
    fwd = distillation_kl_loss(s, t, direction="teacher_forward_kl").item()
    rev = distillation_kl_loss(s, t, direction="reverse_kl").item()
    assert fwd >= 0
    assert rev >= 0
    # Forward and reverse KL are different in general for asymmetric distributions.
    assert abs(fwd - rev) > 1e-5


def test_kl_direction_unknown_raises():
    s = torch.randn(2, 4, 8)
    t = torch.randn(2, 4, 8)
    import pytest
    with pytest.raises(ValueError, match="unknown KL direction"):
        distillation_kl_loss(s, t, direction="psychic_distillation")


def test_phase_aware_teacher_selection():
    from sprl.config import SPRLConfig
    cfg = SPRLConfig()
    cfg.training.distill_teacher_base = "llama_3_1_8b"
    cfg.training.distill_teacher_instruct = "llama_3_1_8b_instruct"
    assert select_teacher_for_phase(cfg, "3a") == "llama_3_1_8b"
    assert select_teacher_for_phase(cfg, "raw_web") == "llama_3_1_8b"
    assert select_teacher_for_phase(cfg, "3c") == "llama_3_1_8b_instruct"
    assert select_teacher_for_phase(cfg, "synthetic") == "llama_3_1_8b_instruct"
