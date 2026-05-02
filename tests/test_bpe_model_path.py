"""BPE forward path: token IDs → backbone → LMHead → [B, S, vocab]."""

import pytest
import torch

from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.training.losses import compute_total_loss


def _bpe_cfg(vocab=512):
    cfg = SPRLConfig(d_model=64, n_layers=2, vocab_size=vocab, max_seq_len_patches=64)
    cfg.tokenizer.type = "bpe"
    cfg.tokenizer.source = "tiny_bpe"
    cfg.vocab_size = vocab
    cfg.patcher.enabled = False
    cfg.attention.mla.n_heads = 4
    cfg.attention.mla.d_c_latent = 32
    cfg.attention.mla.d_qhead = 16
    cfg.attention.mla.d_rope_decoupled = 8
    cfg.moe.num_routed_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 32
    return cfg


def test_bpe_forward_shape():
    cfg = _bpe_cfg(vocab=512)
    # Custom vocab — bypass tokenizer lookup by using a hand-built vocab.
    cfg.tokenizer.source = "tiny_bpe"
    cfg.vocab_size = 32_000  # match registry
    model = SPRLv2(cfg)
    assert model.tokenizer_type == "bpe"
    assert model.token_embed is not None
    assert model.lm_head is not None
    assert model.byte_lm_head is None

    token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model.forward_from_token_ids(token_ids)
    assert out.byte_logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(out.byte_logits).all()


def test_bpe_forward_from_patches_errors():
    """BPE-configured model must refuse the BLT entry point."""
    cfg = _bpe_cfg()
    cfg.vocab_size = 32_000
    model = SPRLv2(cfg)
    with pytest.raises(RuntimeError, match="forward_from_patches is the BLT"):
        model.forward_from_patches(torch.randn(1, 4, cfg.d_model))


def test_blt_forward_from_token_ids_errors():
    """BLT-configured model must refuse the BPE entry point."""
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
    # tokenizer.type defaults to "blt"
    model = SPRLv2(cfg)
    with pytest.raises(RuntimeError, match="forward_from_token_ids is the BPE"):
        model.forward_from_token_ids(torch.randint(0, 256, (1, 4)))


def test_bpe_loss_path_runs():
    cfg = _bpe_cfg()
    cfg.vocab_size = 32_000
    model = SPRLv2(cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    mtp_targets = torch.randint(0, cfg.vocab_size, (2, 8, cfg.mtp.depth))
    out = model.forward_from_token_ids(token_ids, future_targets=mtp_targets)
    loss, parts = compute_total_loss(
        out, byte_targets=token_ids, mtp_targets=mtp_targets
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["lm"] >= 0


def test_bpe_weight_tying_saves_params():
    """Weight-tying ⇒ no separate LM-head parameter matrix."""
    cfg = _bpe_cfg()
    cfg.vocab_size = 32_000

    cfg.tokenizer.weight_tied = True
    tied = SPRLv2(cfg).num_params()

    cfg.tokenizer.weight_tied = False
    untied = SPRLv2(cfg).num_params()

    # Untied should have ~vocab × d_model more params (the extra LM-head matrix).
    diff = untied - tied
    expected = cfg.vocab_size * cfg.d_model
    assert abs(diff - expected) < 1024  # bias-term tolerance


def test_blt_path_unchanged_by_default():
    """Default tokenizer.type='blt' preserves the original byte-level forward."""
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
    assert model.tokenizer_type == "blt"
    assert model.token_embed is None
    assert model.byte_lm_head is not None

    z = torch.randn(1, 4, cfg.d_model)
    out = model.forward_from_patches(z)
    # BLT shape: [B, S, max_patch_bytes, vocab]
    assert out.byte_logits.shape == (1, 4, cfg.patcher.max_patch_bytes, cfg.vocab_size)
