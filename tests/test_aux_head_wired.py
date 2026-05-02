"""AuxiliaryTeacherTokenHead is instantiated when distill_mode demands it,
and the loss path routes KL to its logits, not the main LM head."""

import pytest
import torch

from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.training.losses import compute_total_loss


def _blt_with_aux(distill_enabled: bool, mode: str = "auxiliary_teacher_token_head"):
    cfg = SPRLConfig(d_model=64, n_layers=2, max_seq_len_patches=64)
    cfg.attention.mla.n_heads = 4
    cfg.attention.mla.d_c_latent = 32
    cfg.attention.mla.d_qhead = 16
    cfg.attention.mla.d_rope_decoupled = 8
    cfg.moe.num_routed_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 32
    cfg.patcher.byte_encoder_dim = 32
    cfg.patcher.max_patch_bytes = 8
    cfg.training.distill_enabled = distill_enabled
    cfg.training.distill_mode = mode
    cfg.training.distill_teacher_base = "tiny_bpe"
    cfg.training.distill_teacher_instruct = "tiny_bpe"
    return cfg


def test_aux_head_built_when_mode_set():
    cfg = _blt_with_aux(distill_enabled=True)
    model = SPRLv2(cfg)
    assert model.aux_teacher_head is not None
    # Vocab size matches the registered teacher (tiny_bpe = 32K).
    assert model.aux_teacher_head.teacher_vocab_size == 32_000


def test_aux_head_skipped_when_distill_disabled():
    cfg = _blt_with_aux(distill_enabled=False)
    model = SPRLv2(cfg)
    assert model.aux_teacher_head is None


def test_aux_head_skipped_for_teacher_token_kl_mode():
    """teacher_token_kl distills against the main head, not the aux head."""
    cfg = _blt_with_aux(distill_enabled=True, mode="teacher_token_kl")
    # BLT + teacher_token_kl errors at validate(); skip validate to test the
    # constructor branch only.
    model = SPRLv2(cfg)
    assert model.aux_teacher_head is None


def test_aux_logits_emitted_in_forward():
    cfg = _blt_with_aux(distill_enabled=True)
    model = SPRLv2(cfg)
    z = torch.randn(2, 4, cfg.d_model)
    out = model.forward_from_patches(z)
    assert out.aux_logits is not None
    # Aux head produces [B, S, teacher_vocab]; here z is [B, S_patches, d].
    assert out.aux_logits.shape == (2, 4, 32_000)


def test_loss_uses_aux_logits_when_present():
    """When out.aux_logits is set, KL is computed against aux logits (in
    teacher vocab), not against the main BLT byte logits."""
    cfg = _blt_with_aux(distill_enabled=True)
    model = SPRLv2(cfg)
    z = torch.randn(2, 4, cfg.d_model)
    targets = torch.randint(0, cfg.vocab_size, (2, 4, cfg.patcher.max_patch_bytes))
    teacher_logits = torch.randn(2, 4, 32_000)  # teacher vocab
    out = model.forward_from_patches(z)
    loss, parts = compute_total_loss(
        out, byte_targets=targets, teacher_logits=teacher_logits, distill_kl_weight=0.5
    )
    assert "kl_distill" in parts
    assert torch.isfinite(loss)


def test_loss_errors_on_teacher_vocab_mismatch_without_aux():
    """If the model has NO aux head, teacher_logits in a different vocab
    must be rejected with a clear error."""
    cfg = SPRLConfig(d_model=64, n_layers=2, max_seq_len_patches=64)
    cfg.attention.mla.n_heads = 4
    cfg.attention.mla.d_c_latent = 32
    cfg.attention.mla.d_qhead = 16
    cfg.attention.mla.d_rope_decoupled = 8
    cfg.moe.num_routed_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 32
    cfg.patcher.byte_encoder_dim = 32
    cfg.patcher.max_patch_bytes = 8
    model = SPRLv2(cfg)
    z = torch.randn(1, 4, cfg.d_model)
    targets = torch.randint(0, cfg.vocab_size, (1, 4, cfg.patcher.max_patch_bytes))
    teacher_logits = torch.randn(1, 4, cfg.patcher.max_patch_bytes, 32_000)  # wrong vocab
    out = model.forward_from_patches(z)
    with pytest.raises(ValueError, match="vocab does not match"):
        compute_total_loss(
            out, byte_targets=targets,
            teacher_logits=teacher_logits, distill_kl_weight=0.5,
        )
