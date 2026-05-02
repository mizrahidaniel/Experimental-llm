"""NCA trajectory data source.

Drives `sprl.training.nca_pretrain.GrowingNCA` for a few rollout steps, then
tokenizes each step grid into 32K-vocab codes via the existing
`tokenize_2x2_patches` helper. Stream-friendly: never holds more than one
trajectory in memory.

This module produces *token sequences*, not bytes. It is therefore wired into
the loader as a separate path that bypasses the patcher and feeds the byte
encoder via integer ids cast through the byte vocabulary (mod 256). The cast
preserves the byte-level interface so the rest of the pipeline is unchanged;
it is a deliberate compression of a 32K vocab into 256 buckets and is only
used during the optional NCA pre-pre-training stage.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch

from sprl.training.nca_pretrain import GrowingNCA, tokenize_2x2_patches


def nca_trajectory_stream(
    *,
    seed: int = 0,
    grid_size: int = 16,
    n_channels: int = 16,
    hidden: int = 64,
    n_steps: int = 8,
    vocab_size: int = 32768,
    max_examples: Optional[int] = None,
) -> Iterator[bytes]:
    """Yield byte streams (post mod-256 cast) usable by the rest of the pipeline."""
    g = torch.Generator().manual_seed(seed)
    nca = GrowingNCA(n_channels=n_channels, hidden=hidden)
    nca.eval()
    n = 0
    while True:
        if max_examples is not None and n >= max_examples:
            return
        state = torch.zeros(1, n_channels, grid_size, grid_size)
        state[0, 3, grid_size // 2, grid_size // 2] = 1.0
        state = state + 0.01 * torch.randn(state.shape, generator=g)
        with torch.no_grad():
            traj = nca(state, n_steps=n_steps)
        all_tokens: list[torch.Tensor] = []
        for step_state in traj[1:]:
            tok = tokenize_2x2_patches(step_state[0], vocab_size=vocab_size)
            all_tokens.append(tok)
        codes = torch.cat(all_tokens, dim=0)  # [N_tokens]
        bytes_out = (codes % 256).to(torch.uint8).cpu().numpy().tobytes()
        n += 1
        yield bytes_out


def nca_token_stream(
    *,
    seed: int = 0,
    grid_size: int = 16,
    n_channels: int = 16,
    hidden: int = 64,
    n_steps: int = 8,
    vocab_size: int = 32768,
    max_examples: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    """Variant that yields the raw `[N_tokens]` int tensor (no byte cast).

    Useful for direct NCA pre-pre-training loops that want the full 32K vocab.
    """
    g = torch.Generator().manual_seed(seed)
    nca = GrowingNCA(n_channels=n_channels, hidden=hidden)
    nca.eval()
    n = 0
    while True:
        if max_examples is not None and n >= max_examples:
            return
        state = torch.zeros(1, n_channels, grid_size, grid_size)
        state[0, 3, grid_size // 2, grid_size // 2] = 1.0
        state = state + 0.01 * torch.randn(state.shape, generator=g)
        with torch.no_grad():
            traj = nca(state, n_steps=n_steps)
        toks = []
        for step_state in traj[1:]:
            toks.append(tokenize_2x2_patches(step_state[0], vocab_size=vocab_size))
        yield torch.cat(toks, dim=0)
        n += 1
