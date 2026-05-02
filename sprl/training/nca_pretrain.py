"""NCA pre-pre-training data generator (Section 4.2).

Generates trajectories from a small Mordvintsev-style growing neural CA on a
2D grid, tokenizes 2x2 patches into a 32K vocab, and yields next-token
prediction examples. Use as a 100-200M-token "free win" before language
pretraining.

Tokenization: prefer the VQ codebook (`tokenize_2x2_patches_vq`); the original
LSH (`tokenize_2x2_patches_lsh`) is kept as a fallback for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GrowingNCA(nn.Module):
    """16-channel 2D NCA, sobel filters + perceive + dx via 1x1 conv stack.

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
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1.0
        kernel = torch.stack([identity, sobel_x, sobel_y]).unsqueeze(1)
        kernel = kernel.repeat(n_channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def perceive(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x_r = x.reshape(b * c, 1, h, w)
        y = F.conv2d(x_r, self.kernel[:3], padding=1)
        return y.reshape(b, c * 3, h, w)

    def forward(self, x: Tensor, n_steps: int) -> list[Tensor]:
        traj = [x]
        for _ in range(n_steps):
            dx = self.update(self.perceive(x))
            mask = (torch.rand_like(x[:, :1]) < self.fire_rate).float()
            x = x + dx * mask
            traj.append(x)
        return traj


# ---------------------------------------------------------------------------
# Patch flattening (shared by both tokenizers)

def _flatten_2x2_patches(state: Tensor) -> Tensor:
    """[C, H, W] -> [N_tokens, 4*C] row-major over 2x2 patches."""
    C, H, W = state.shape
    h2 = (H // 2) * 2
    w2 = (W // 2) * 2
    s = state[:, :h2, :w2].reshape(C, h2 // 2, 2, w2 // 2, 2)
    s = s.permute(1, 3, 0, 2, 4).reshape((h2 // 2) * (w2 // 2), C * 2 * 2)
    return s.contiguous()


# ---------------------------------------------------------------------------
# LSH (legacy fallback)

def tokenize_2x2_patches_lsh(state: Tensor, vocab_size: int = 32768) -> Tensor:
    """Hash-based 2x2 patch tokenizer (original implementation, kept as fallback).

    Non-deterministic w.r.t. norm changes -- prefer `tokenize_2x2_patches_vq`.
    """
    s = _flatten_2x2_patches(state)
    signs = (s > 0).int()
    weights = torch.tensor([2 ** i for i in range(s.shape[1])], device=s.device)
    codes = (signs * weights[: s.shape[1]]).sum(dim=-1) % vocab_size
    return codes


# Backwards-compat alias for callers that imported the old name.
tokenize_2x2_patches = tokenize_2x2_patches_lsh


# ---------------------------------------------------------------------------
# VQ codebook (preferred)

@dataclass
class VQConfig:
    vocab_size: int = 32768
    code_dim: int = 64  # = 4 * C with C = 16
    decay: float = 0.99
    eps: float = 1e-5
    init_scale: float = 0.1


class VQCodebook(nn.Module):
    """L2-normalized VQ codebook with EMA centroid updates.

    Following arXiv 2106.13409 Section 3 (VQ-VAE training rule). Codes live on
    the unit sphere (L2-normalized); assignment uses cosine similarity, which
    is equivalent to nearest-Euclidean on normalized vectors.

    EMA update (per batch):
        N_k <- decay * N_k + (1 - decay) * count_k
        m_k <- decay * m_k + (1 - decay) * sum_{x in cluster k} x
        e_k <- m_k / (N_k + eps)        (then renormalized to the sphere)

    `quantize(x)` returns `(z_q, indices, commit_loss)` with straight-through
    gradient w.r.t. the encoder input.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        code_dim: int = 64,
        decay: float = 0.99,
        eps: float = 1e-5,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.code_dim = code_dim
        self.decay = float(decay)
        self.eps = float(eps)

        codes = torch.randn(vocab_size, code_dim) * init_scale
        codes = F.normalize(codes, dim=-1)
        self.register_buffer("codes", codes)
        self.register_buffer("ema_count", torch.zeros(vocab_size))
        self.register_buffer("ema_sum", codes.clone())
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.int32))

    @torch.no_grad()
    def _bootstrap_init(self, x: Tensor) -> None:
        """Seed the codebook from the first batch (random sampling)."""
        N = x.shape[0]
        if N == 0:
            return
        idx = torch.randint(0, N, (self.vocab_size,), device=x.device)
        sample = F.normalize(x[idx], dim=-1)
        self.codes.copy_(sample)
        self.ema_sum.copy_(sample)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(1)

    def assign(self, x: Tensor) -> Tensor:
        """x: [N, code_dim] (will be L2-normalized). Returns indices [N]."""
        xn = F.normalize(x, dim=-1)
        sim = xn @ self.codes.t()
        return sim.argmax(dim=-1)

    @torch.no_grad()
    def ema_update(self, x: Tensor, indices: Tensor) -> None:
        """EMA update of codebook from a batch (no grad)."""
        xn = F.normalize(x, dim=-1)
        V = self.vocab_size
        onehot = F.one_hot(indices, num_classes=V).to(xn.dtype)
        count = onehot.sum(dim=0)
        sum_x = onehot.t() @ xn
        self.ema_count.mul_(self.decay).add_(count, alpha=(1 - self.decay))
        self.ema_sum.mul_(self.decay).add_(sum_x, alpha=(1 - self.decay))
        n = self.ema_count.sum()
        smoothed = (self.ema_count + self.eps) / (n + V * self.eps) * n
        new_codes = self.ema_sum / smoothed.unsqueeze(-1).clamp_min(self.eps)
        self.codes.copy_(F.normalize(new_codes, dim=-1))

    def forward(self, x: Tensor) -> Tensor:
        """Convenience: returns indices."""
        return self.assign(x)

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize x [N, code_dim] -> (z_q, indices, commit_loss).

        Straight-through estimator: dL/dx = dL/dz_q.
        Commitment loss: ||x - sg(z_q)||^2 (on L2-normalized x).
        Codebook is updated in-place via EMA when in training mode.
        """
        if int(self.initialized.item()) == 0 and self.training:
            self._bootstrap_init(x)
        idx = self.assign(x)
        z_q = self.codes[idx]
        if self.training:
            self.ema_update(x.detach(), idx.detach())
        xn = F.normalize(x, dim=-1)
        commit = (xn - z_q.detach()).pow(2).sum(dim=-1).mean()
        z_q_st = xn + (z_q - xn).detach()
        return z_q_st, idx, commit

    def embed(self, indices: Tensor) -> Tensor:
        return self.codes[indices]


def tokenize_2x2_patches_vq(
    state: Tensor,
    codebook: VQCodebook,
    vocab_size: int | None = None,  # noqa: ARG001 - kept for API parity
) -> Tensor:
    """Deterministic VQ tokenizer for 2x2 patches of a [C, H, W] state.

    `vocab_size` is ignored (kept for API parity with the LSH variant); the
    codebook determines the vocabulary. Returns [N_tokens] long indices.
    """
    s = _flatten_2x2_patches(state)
    if s.shape[-1] != codebook.code_dim:
        raise ValueError(
            f"patch dim {s.shape[-1]} != codebook.code_dim {codebook.code_dim}"
        )
    with torch.no_grad():
        return codebook.assign(s)
