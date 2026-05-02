"""NCA pre-pre-training data generator (Section 4.2).

Generates trajectories from a small Mordvintsev-style growing neural CA on a
2D grid, tokenizes 2×2 patches into a 32K vocab, and yields next-token
prediction examples. Use as a 100–200M-token "free win" before language
pretraining.

This module gives a *reference* CA + tokenizer; you should replace it with a
pretrained CA checkpoint for serious runs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GrowingNCA(nn.Module):
    """16-channel 2D NCA, sobel filters + perceive + dx via 1×1 conv stack.

    See https://distill.pub/2020/growing-ca/ for the original formulation.
    """

    def __init__(self, n_channels: int = 16, hidden: int = 64, fire_rate: float = 0.5):
        super().__init__()
        self.n = n_channels
        self.fire_rate = fire_rate
        self.update = nn.Sequential(
            nn.Conv2d(3 * n_channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, n_channels, 1),
        )
        # Sobel filters as fixed conv weights.
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1.0
        kernel = torch.stack([identity, sobel_x, sobel_y]).unsqueeze(1)  # [3, 1, 3, 3]
        kernel = kernel.repeat(n_channels, 1, 1, 1)  # depthwise
        self.register_buffer("kernel", kernel)

    def perceive(self, x: Tensor) -> Tensor:
        # Depthwise conv.
        b, c, h, w = x.shape
        x_r = x.reshape(b * c, 1, h, w)
        y = F.conv2d(x_r, self.kernel[:3], padding=1)  # [b*c, 3, h, w]
        return y.reshape(b, c * 3, h, w)

    def forward(self, x: Tensor, n_steps: int) -> list[Tensor]:
        """x: [B, C, H, W]. Returns trajectory [n_steps+1] of states."""
        traj = [x]
        for _ in range(n_steps):
            dx = self.update(self.perceive(x))
            mask = (torch.rand_like(x[:, :1]) < self.fire_rate).float()
            x = x + dx * mask
            traj.append(x)
        return traj


def tokenize_2x2_patches(state: Tensor, vocab_size: int = 32768) -> Tensor:
    """Tokenize 2×2 patches of a [C, H, W] state into discrete codes.

    This reference uses a hash-based codebook (LSH); for a serious run, swap in
    a proper VQ codebook or a learned scalar quantizer.
    """
    C, H, W = state.shape
    h2 = (H // 2) * 2
    w2 = (W // 2) * 2
    s = state[:, :h2, :w2].reshape(C, h2 // 2, 2, w2 // 2, 2)
    s = s.permute(1, 3, 0, 2, 4).reshape((h2 // 2) * (w2 // 2), C * 2 * 2)
    # Quantize each row to a fixed sign-bit signature, hash to vocab_size.
    signs = (s > 0).int()
    weights = torch.tensor([2 ** i for i in range(s.shape[1])], device=s.device)
    codes = (signs * weights[: s.shape[1]]).sum(dim=-1) % vocab_size
    return codes  # [N_tokens]
