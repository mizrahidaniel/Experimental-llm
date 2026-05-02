"""TokenLevelDRAMBlock: per-token K* mask correctness + freezing invariant."""

import torch
from torch import nn

from sprl.recurrent.dram_block import DRAMBlock, TokenLevelDRAMBlock


def _toy_attn(d):
    return nn.Linear(d, d, bias=False)


def _toy_ffn(d):
    return nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))


def test_token_level_matches_constant_k_case():
    """When all tokens have the same K, TokenLevel must match DRAMBlock."""
    torch.manual_seed(0)
    d = 8
    block_const = DRAMBlock(
        d_model=d,
        attention_module=_toy_attn(d),
        ffn_module=_toy_ffn(d),
        use_depth_attention=False,
    )
    block_token = TokenLevelDRAMBlock(
        d_model=d,
        attention_module=block_const.attn,
        ffn_module=block_const.ffn,
        use_depth_attention=False,
    )
    # Tie internal modules.
    block_token.norm1 = block_const.norm1
    block_token.norm2 = block_const.norm2

    z0 = torch.randn(1, 5, d)
    n_iter = 3
    z_const, _ = block_const(z0, n_iter=n_iter, record_history=False)
    k = torch.full((1, 5), n_iter, dtype=torch.long)
    z_token, _ = block_token.forward_token_level(z0, k)
    assert torch.allclose(z_const, z_token, atol=1e-5)


def test_token_level_freezes_done_tokens():
    """Tokens with K=1 should equal the result of one block step; tokens
    with K=3 should change relative to K=1."""
    torch.manual_seed(0)
    d = 8
    block = TokenLevelDRAMBlock(
        d_model=d,
        attention_module=_toy_attn(d),
        ffn_module=_toy_ffn(d),
        use_depth_attention=False,
    )
    z0 = torch.randn(1, 4, d)
    k = torch.tensor([[1, 1, 3, 3]])
    z, _ = block.forward_token_level(z0, k)

    # Re-run with k=1 everywhere — first two positions should match.
    z_k1, _ = block.forward_token_level(z0, torch.full((1, 4), 1, dtype=torch.long))
    assert torch.allclose(z[0, 0], z_k1[0, 0], atol=1e-6)
    assert torch.allclose(z[0, 1], z_k1[0, 1], atol=1e-6)
    # And positions 2, 3 should NOT match z_k1 (more iterations).
    assert not torch.allclose(z[0, 2], z_k1[0, 2], atol=1e-6)
    assert not torch.allclose(z[0, 3], z_k1[0, 3], atol=1e-6)


def test_token_level_max_iter_cap():
    """max_iter_cap should bound the number of iterations executed."""
    torch.manual_seed(0)
    d = 8
    block = TokenLevelDRAMBlock(
        d_model=d,
        attention_module=_toy_attn(d),
        ffn_module=_toy_ffn(d),
        use_depth_attention=False,
    )
    z0 = torch.randn(1, 3, d)
    # Ask for 5 iterations everywhere, but cap at 2.
    k = torch.full((1, 3), 5, dtype=torch.long)
    z_capped, _ = block.forward_token_level(z0, k, max_iter_cap=2)

    # Should match k=2 everywhere.
    z_k2, _ = block.forward_token_level(z0, torch.full((1, 3), 2, dtype=torch.long))
    assert torch.allclose(z_capped, z_k2, atol=1e-6)


def test_token_level_grad_flows_with_depth_attention():
    d = 8
    block = TokenLevelDRAMBlock(
        d_model=d,
        attention_module=_toy_attn(d),
        ffn_module=_toy_ffn(d),
        use_depth_attention=True,
    )
    z0 = torch.randn(1, 4, d, requires_grad=True)
    k = torch.tensor([[2, 3, 1, 2]])
    z, _ = block.forward_token_level(z0, k, record_history=True)
    z.sum().backward()
    assert torch.isfinite(z0.grad).all()


def test_token_level_handles_float_k_per_token():
    """If k_per_token comes from an STE (float), the block should still work."""
    d = 8
    block = TokenLevelDRAMBlock(
        d_model=d,
        attention_module=_toy_attn(d),
        ffn_module=_toy_ffn(d),
        use_depth_attention=False,
    )
    z0 = torch.randn(1, 4, d)
    k_float = torch.tensor([[1.0, 2.0, 2.0, 3.0]])
    z, _ = block.forward_token_level(z0, k_float)
    assert torch.isfinite(z).all()
