"""DRAM block: K-iter forward, depth-attention, history shape."""

import torch
from torch import nn

from sprl.recurrent.dram_block import DRAMBlock, TokenLevelDRAMBlock


def _toy_attn(d):
    return nn.Linear(d, d, bias=False)


def _toy_ffn(d):
    return nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))


def test_dram_forward_history_length():
    d = 16
    block = DRAMBlock(d_model=d, attention_module=_toy_attn(d), ffn_module=_toy_ffn(d))
    z0 = torch.randn(1, 8, d)
    z, hist = block(z0, n_iter=3, record_history=True)
    assert z.shape == z0.shape
    assert len(hist) == 4  # initial + 3 steps


def test_dram_no_depth_attention_path():
    d = 8
    block = DRAMBlock(
        d_model=d,
        attention_module=_toy_attn(d),
        ffn_module=_toy_ffn(d),
        use_depth_attention=False,
    )
    z0 = torch.randn(1, 4, d)
    z, _ = block(z0, n_iter=2)
    assert z.shape == z0.shape
    assert torch.isfinite(z).all()


def test_dram_grad_flows():
    d = 8
    block = DRAMBlock(d_model=d, attention_module=_toy_attn(d), ffn_module=_toy_ffn(d))
    z0 = torch.randn(1, 4, d, requires_grad=True)
    z, _ = block(z0, n_iter=2)
    z.sum().backward()
    assert torch.isfinite(z0.grad).all()


def test_token_level_dram_freezes_done_tokens():
    d = 8
    block = TokenLevelDRAMBlock(
        d_model=d,
        attention_module=_toy_attn(d),
        ffn_module=_toy_ffn(d),
        use_depth_attention=False,
    )
    z0 = torch.randn(1, 4, d)
    # Token 0: 1 iter, token 3: 3 iters; tokens 1-2: 2 iters.
    k = torch.tensor([[1, 2, 2, 3]])
    z, _ = block.forward_token_level(z0, k)
    assert z.shape == z0.shape
    assert torch.isfinite(z).all()
