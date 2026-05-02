"""RecurrentMiddleBlock + middle-cell layout in SPRLv2."""

import torch
from torch import nn

from sprl.config import SPRLConfig
from sprl.model import SPRLv2
from sprl.recurrent import RecurrentMiddleBlock


def _toy_layer(d):
    """Identity-ish layer with a small linear so iteration changes the state."""
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))


def test_middle_block_shape_and_iteration():
    d = 8
    block = RecurrentMiddleBlock(
        prefix_layers=[_toy_layer(d)],
        cell_layers=[_toy_layer(d), _toy_layer(d)],
        suffix_layers=[_toy_layer(d)],
    )
    z = torch.randn(2, 4, d)
    k = torch.tensor([1, 3])  # batch element 0 iterates once; element 1 iterates 3
    out, history = block(z, k_per_block=k, return_iterations=True)
    assert out.shape == z.shape
    # History has max_K + 1 entries.
    assert len(history) == 4


def test_middle_block_freeze_after_K():
    """Element iterates K times then freezes; further iterations should
    leave its state unchanged."""
    d = 8
    cell = [_toy_layer(d)]
    block = RecurrentMiddleBlock([], cell, [])
    z = torch.randn(2, 4, d)
    # Both elements iterate K=2.
    out_k2, _ = block(z, k_per_block=torch.tensor([2, 2]))
    # Now batch element 0 iterates 2, element 1 iterates 5.
    out_mixed, _ = block(z, k_per_block=torch.tensor([2, 5]))
    # Element 0 in both runs should get exactly 2 cell iterations ⇒ identical.
    assert torch.allclose(out_k2[0], out_mixed[0], atol=1e-5)


def test_sprlv2_middle_cell_branch_runs():
    cfg = SPRLConfig(d_model=64, n_layers=8, max_seq_len_patches=64, vocab_size=256)
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
    cfg.dram.cell_type = "recurrent_middle_block"
    cfg.dram.prefix_layers = 2
    cfg.dram.recurrent_cell_layers = 4
    cfg.dram.suffix_layers = 2
    cfg.dram.routing_granularity = "per_block"
    cfg.dram.use_depth_attention = False

    model = SPRLv2(cfg)
    z = torch.randn(2, 8, 64)
    out = model.forward_from_patches(z)
    assert out.byte_logits.shape == (2, 8, cfg.patcher.max_patch_bytes, cfg.vocab_size)
    assert torch.isfinite(out.byte_logits).all()
    # Recurrence is in middle.cell, not in self.layers.
    assert len(model.layers) == 0
    assert model.middle is not None
    assert len(model.middle.prefix) == 2
    assert len(model.middle.cell) == 4
    assert len(model.middle.suffix) == 2


def test_sprlv2_uniform_layer_legacy_path_still_works():
    """Default cell_type=uniform_layers preserves the legacy path."""
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
    # cell_type defaults to "uniform_layers"

    model = SPRLv2(cfg)
    assert model.middle is None
    assert len(model.layers) == 2
    z = torch.randn(2, 8, 64)
    out = model.forward_from_patches(z)
    assert torch.isfinite(out.byte_logits).all()


def test_middle_block_grad_flows():
    d = 8
    block = RecurrentMiddleBlock(
        prefix_layers=[_toy_layer(d)],
        cell_layers=[_toy_layer(d), _toy_layer(d)],
        suffix_layers=[_toy_layer(d)],
    )
    z = torch.randn(2, 4, d, requires_grad=True)
    out, _ = block(z, k_per_block=torch.tensor([2, 3]))
    out.sum().backward()
    assert torch.isfinite(z.grad).all()
