"""GaLore: gradient low-rank projection.

Per https://github.com/jiaweizzhao/GaLore. Halves optimizer state on the
largest MoE expert weights without sacrificing convergence.

This module provides a self-contained projector. The actual optimizer wrapping
is left to a thin shim in `training/optim.py` so callers can swap in
`bitsandbytes.optim.AdamW8bit` cleanly when bitsandbytes is available.
"""

from __future__ import annotations

import torch
from torch import Tensor


class GaLoreProjector:
    """Projects a 2D gradient onto a rank-r subspace; un-projects after the optim step.

    Subspace is re-estimated every `update_proj_every` steps via a randomized SVD.
    """

    def __init__(self, rank: int = 128, update_proj_every: int = 200):
        self.rank = rank
        self.update_proj_every = update_proj_every
        self._step = 0
        self._P: Tensor | None = None  # [rank, M]
        self._side = "right"  # which side to project (chosen lazily)

    def _maybe_update_basis(self, g: Tensor) -> None:
        if self._P is not None and self._step % self.update_proj_every != 0:
            return
        rows, cols = g.shape
        if cols >= rows:
            U, _, _ = torch.linalg.svd(g, full_matrices=False)
            self._P = U[:, : self.rank].t().contiguous()  # [rank, rows]
            self._side = "left"
        else:
            _, _, Vh = torch.linalg.svd(g, full_matrices=False)
            self._P = Vh[: self.rank, :].contiguous()  # [rank, cols]
            self._side = "right"

    def project(self, g: Tensor) -> Tensor:
        assert g.dim() == 2
        self._maybe_update_basis(g)
        if self._side == "right":
            return g @ self._P.t()  # [rows, rank]
        return self._P @ g  # [rank, cols]

    def unproject(self, g_low: Tensor) -> Tensor:
        if self._side == "right":
            return g_low @ self._P  # [rows, cols]
        return self._P.t() @ g_low

    def step(self) -> None:
        self._step += 1
