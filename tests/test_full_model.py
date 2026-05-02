"""End-to-end smoke test: build SPRLv2 from a tiny config, drive forward + backward."""

import torch

from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.training.losses import compute_total_loss


def _tiny_cfg(**overrides) -> SPRLConfig:
    cfg = SPRLConfig(
        d_model=64,
        n_layers=2,
        max_seq_len_patches=64,
    )
    cfg.attention.mla.n_heads = 4
    cfg.attention.mla.d_c_latent = 32
    cfg.attention.mla.d_qhead = 16
    cfg.attention.mla.d_rope_decoupled = 8
    cfg.attention.dsa.dense_until_seqlen = 256  # always dense at this scale
    cfg.attention.sliding.window_size = 32
    cfg.moe.num_routed_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.active_per_token = 2
    cfg.moe.expert_dim = 32
    cfg.patcher.byte_encoder_dim = 32
    cfg.patcher.max_patch_bytes = 8
    cfg.mtp.depth = 2
    for k, v in overrides.items():
        head = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            head = getattr(head, p)
        setattr(head, parts[-1], v)
    return cfg


def test_build_and_forward_baseline():
    cfg = _tiny_cfg()
    model = SPRLv2(cfg)
    z = torch.randn(1, 8, cfg.d_model)
    out = model.forward_from_patches(
        z,
        future_targets=torch.randint(0, cfg.vocab_size, (1, 8, cfg.mtp.depth)),
    )
    assert out.byte_logits.shape == (1, 8, cfg.patcher.max_patch_bytes, cfg.vocab_size)
    assert torch.isfinite(out.byte_logits).all()
    assert "mean_k" in out.diagnostics


def test_build_and_forward_with_kalman():
    cfg = _tiny_cfg(**{"kalman.enabled": True, "dram.router.enabled": True})
    model = SPRLv2(cfg)
    z = torch.randn(1, 8, cfg.d_model)
    out = model.forward_from_patches(z)
    assert "log_det_Lambda_mean" in out.diagnostics


def test_build_and_forward_with_tropical():
    cfg = _tiny_cfg(**{"attention.tropical.enabled": True})
    model = SPRLv2(cfg)
    z = torch.randn(1, 8, cfg.d_model)
    out = model.forward_from_patches(z)
    assert torch.isfinite(out.byte_logits).all()


def test_build_and_forward_with_rg_flow():
    cfg = _tiny_cfg(**{"dram.rg_flow.enabled": True})
    model = SPRLv2(cfg)
    z = torch.randn(1, 8, cfg.d_model)
    out = model.forward_from_patches(z)
    assert "rg_flow" in out.aux_losses
    assert "T_b_norm" in out.diagnostics


def test_loss_backward_runs_baseline():
    cfg = _tiny_cfg()
    model = SPRLv2(cfg)
    z = torch.randn(1, 8, cfg.d_model)
    targets = torch.randint(0, cfg.vocab_size, (1, 8, cfg.patcher.max_patch_bytes))
    mtp_targets = torch.randint(0, cfg.vocab_size, (1, 8, cfg.mtp.depth))
    out = model.forward_from_patches(z, future_targets=mtp_targets)
    loss, parts = compute_total_loss(
        out,
        byte_targets=targets,
        mtp_targets=mtp_targets,
    )
    loss.backward()
    grads_finite = all(
        (p.grad is None) or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert grads_finite
