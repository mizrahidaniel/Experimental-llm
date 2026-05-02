"""sequence_level distillation: no logit storage, just CE on teacher tokens."""

import pytest
import torch

from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.training.losses import compute_total_loss


def test_sequence_level_validates_without_topk_storage():
    """Unlike teacher_token_kl, sequence_level needs no logit storage."""
    cfg = SPRLConfig(variant="legacy")
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "sequence_level"
    cfg.training.distill_store_indices = False
    cfg.training.distill_store_logits = False
    cfg.training.distill_store_teacher_logsumexp = False
    cfg.validate()  # no raise


def test_sequence_level_compatible_with_blt():
    """The whole point of sequence_level is sidestepping vocab alignment."""
    cfg = SPRLConfig(variant="legacy")
    cfg.patcher.enabled = True
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "sequence_level"
    cfg.validate()  # no raise — BLT + sequence_level is OK


def test_unknown_distill_mode_errors():
    cfg = SPRLConfig(variant="legacy")
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "psychic_distillation"
    with pytest.raises(ValueError, match="distill_mode"):
        cfg.validate()


def test_sequence_level_loss_is_plain_ce():
    """sequence_level training is just CE — pass `teacher_logits=None` and
    `distill_kl_weight=0` and the loss reduces to the LM head's CE."""
    cfg = SPRLConfig(d_model=64, n_layers=2, vocab_size=32_000, max_seq_len_patches=64)
    cfg.tokenizer.type = "bpe"
    cfg.tokenizer.source = "tiny_bpe"
    cfg.patcher.enabled = False
    cfg.attention.mla.n_heads = 4
    cfg.attention.mla.d_c_latent = 32
    cfg.attention.mla.d_qhead = 16
    cfg.attention.mla.d_rope_decoupled = 8
    cfg.moe.num_routed_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 32
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "sequence_level"
    cfg.training.distill_store_indices = False
    cfg.training.distill_store_logits = False
    cfg.training.distill_store_teacher_logsumexp = False
    cfg.validate()

    model = SPRLv2(cfg)
    # Sequence-level distillation: the training data is teacher-generated text
    # tokenized with the teacher's tokenizer; the training loop sees them as
    # ground-truth targets.
    teacher_token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model.forward_from_token_ids(teacher_token_ids)
    loss, parts = compute_total_loss(
        out,
        byte_targets=teacher_token_ids,
        teacher_logits=None,
        distill_kl_weight=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "kl_distill" not in parts
    assert parts["lm"] >= 0
