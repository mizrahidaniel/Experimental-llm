"""v3.1 recurrent middle-cell architecture (Geiping/Huginn-style).

Layout:
    input -> prefix_layers (no recurrence)
          -> recurrent_cell_layers (TIED weights, iterated K times)
          -> suffix_layers (no recurrence)
          -> output

Routing is per-block: K is one integer per batch element (per "block"). All
tokens in the same block iterate K times together. Per-token routing is
forbidden under attention because it leaves the attended-to-tokens at
unequal iteration depth — see spec §4.5.2.

The cell layers are *shared* across iterations: a single
`nn.ModuleList[layer_0, ..., layer_{cell_depth-1}]` is run K times. Iteration
freezing is implemented with a max-K masked update so the whole batch can
share the same forward graph.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor, nn


class RecurrentMiddleBlock(nn.Module):
    """Composes prefix + tied cell × K + suffix, with per-block iteration K.

    Args:
        prefix_layers: list of nn.Modules (each consumes [B,S,D] and returns [B,S,D]).
        cell_layers:   list of nn.Modules (the *tied* recurrent cell — these
                        are reapplied at each iteration).
        suffix_layers: list of nn.Modules.
    """

    def __init__(
        self,
        prefix_layers: Sequence[nn.Module],
        cell_layers: Sequence[nn.Module],
        suffix_layers: Sequence[nn.Module],
    ):
        super().__init__()
        self.prefix = nn.ModuleList(list(prefix_layers))
        self.cell = nn.ModuleList(list(cell_layers))
        self.suffix = nn.ModuleList(list(suffix_layers))

    @property
    def n_cell_layers(self) -> int:
        return len(self.cell)

    def forward(
        self,
        z: Tensor,
        k_per_block: Tensor,
        return_iterations: bool = False,
    ) -> tuple[Tensor, List[Tensor]]:
        """Args:
            z: [B, S, D] input.
            k_per_block: [B] int tensor — iteration count per batch element.
            return_iterations: if True, also return the list of cell-iteration
                outputs (used by RG-flow diagnostic).

        Returns:
            z_out: [B, S, D] after prefix + cell·K + suffix.
            iter_outputs: list of length max_K+1 of [B, S, D] cell outputs
                          (only when return_iterations=True; else empty).
        """
        # --- Prefix (non-recurrent) --------------------------------------
        for layer in self.prefix:
            z = layer(z)

        # --- Recurrent cell, iterated max(K) times with per-block masking
        if k_per_block.dtype.is_floating_point:
            max_k = int(k_per_block.detach().max().item())
        else:
            max_k = int(k_per_block.max().item())
        max_k = max(1, max_k)

        iter_outputs: List[Tensor] = []
        if return_iterations:
            iter_outputs.append(z)

        z_iter = z
        for k in range(max_k):
            # Mask: 1 for batch elements still iterating, 0 for frozen.
            active = (k_per_block > k).to(z_iter.dtype).view(-1, 1, 1)
            z_new = z_iter
            for layer in self.cell:
                z_new = layer(z_new)
            z_iter = z_new * active + z_iter * (1.0 - active)
            if return_iterations:
                iter_outputs.append(z_iter)
        z = z_iter

        # --- Suffix (non-recurrent) --------------------------------------
        for layer in self.suffix:
            z = layer(z)

        return z, iter_outputs
