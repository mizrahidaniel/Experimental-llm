"""Kalman passive_probe must fuse output via a learnable gated residual."""

import torch

from sprl.config import SPRLConfig
from sprl.model import SPRLv2


def _tiny_cfg():
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
    cfg.kalman.enabled = True
    cfg.kalman.rank = 4
    cfg.kalman.mode = "passive_probe"
    cfg.kalman.fusion_gate_init = -3.0
    return cfg


def test_passive_probe_creates_fusion_gate():
    model = SPRLv2(_tiny_cfg())
    assert model.kalman_fusion_gate is not None
    assert torch.isclose(model.kalman_fusion_gate, torch.tensor(-3.0))


def test_passive_probe_fuses_kalman_output():
    model = SPRLv2(_tiny_cfg())
    z = torch.randn(2, 8, 64)
    out = model.forward_from_patches(z)
    # Diagnostic logged.
    assert "kalman_fusion_gate" in out.diagnostics
    expected_sigmoid = float(torch.sigmoid(torch.tensor(-3.0)).item())
    assert abs(out.diagnostics["kalman_fusion_gate"] - expected_sigmoid) < 1e-5


def test_diagnostic_probe_skips_fusion():
    """In `diagnostic_probe` mode the gate is not applied to the hidden stream
    (Kalman output is logged but kept off the residual)."""
    cfg = _tiny_cfg()
    cfg.kalman.mode = "diagnostic_probe"
    model = SPRLv2(cfg)
    assert model.kalman_diagnostic_only

    z = torch.randn(2, 8, 64)
    out = model.forward_from_patches(z)
    # Still logs.
    assert "log_det_Lambda_mean" in out.diagnostics


def test_fusion_gate_receives_gradients():
    """The gate parameter must accumulate gradient through the fused residual."""
    model = SPRLv2(_tiny_cfg())
    z = torch.randn(2, 8, 64)
    out = model.forward_from_patches(z)
    loss = out.byte_logits.float().mean()
    loss.backward()
    assert model.kalman_fusion_gate.grad is not None
    assert torch.isfinite(model.kalman_fusion_gate.grad).all()
    # Non-zero w.h.p. after a single backward through fp ops.
    assert model.kalman_fusion_gate.grad.abs().item() > 0
