"""SPRL-v2 top-level model.

Assembles every component:

  bytes -> EntropyPatcher -> ByteEncoder -> [N_patches, d_model]
        -> KalmanInfoMemory (optional, Bet A)
        -> for each layer in n_layers:
              DRAMBlock with K iterations (depth-recurrent)
                inner step = SPRLLayer (MLA/sliding + FFN/MoE)
                tropical heads optionally spliced into MLA (Bet B)
                RG-flow regularizer collected over iterations (Bet C)
        -> ByteLMHead -> per-byte logits
        -> MTPHead -> next-2-byte logits

For the pilot config, depth recurrence is applied at the *macro* layer level:
each layer is iterated K* times. The outer layer stack still has n_layers
distinct (untied) layer modules, but within a layer the block is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sprl.blocks import SPRLLayer, make_attention, make_ffn
from sprl.config import SPRLConfig
from sprl.heads import ByteLMHead, MultiTokenPredictionHead
from sprl.memory import KalmanInfoMemory
from sprl.patcher import ByteEncoder, ByteLM, EntropyPatcher
from sprl.recurrent import (
    ActiveInferenceRouter,
    DRAMBlock,
    EntropyRouter,
    RecurrentMiddleBlock,
    RGFlowRegularizer,
    TokenLevelDRAMBlock,
)


@dataclass
class SPRLOutput:
    byte_logits: Tensor  # [N_patches, n_bytes_per_patch, vocab]
    mtp_logits: Optional[List[Tensor]]
    aux_losses: dict
    diagnostics: dict


class SPRLv2(nn.Module):
    def __init__(self, cfg: SPRLConfig):
        super().__init__()
        self.cfg = cfg

        # ----- patcher (encoder side) ---------------------------------
        if cfg.patcher.enabled:
            self.byte_lm = ByteLM(
                dim=cfg.patcher.byte_lm_dim, n_layers=cfg.patcher.byte_lm_layers
            )
            self.patcher = EntropyPatcher(
                self.byte_lm,
                threshold=cfg.patcher.entropy_threshold,
                min_patch_bytes=cfg.patcher.min_patch_bytes,
                max_patch_bytes=cfg.patcher.max_patch_bytes,
            )
            self.byte_encoder = ByteEncoder(
                byte_dim=cfg.patcher.byte_encoder_dim,
                patch_dim=cfg.d_model,
                n_layers=cfg.patcher.byte_encoder_layers,
                max_patch_bytes=cfg.patcher.max_patch_bytes,
            )
        else:
            self.byte_lm = None
            self.patcher = None
            self.byte_encoder = None

        # ----- Bet A: Kalman info memory ------------------------------
        if cfg.kalman.enabled:
            self.kalman = KalmanInfoMemory(
                d_model=cfg.d_model,
                rank=cfg.kalman.rank,
                eps_diag=cfg.kalman.eps_diag,
                chunk_size=cfg.kalman.chunk_size,
            )
            # v3.1: gated residual fusion. `passive_probe` mode produces an
            # output path so the gate gets LM gradients; `diagnostic_probe`
            # zeros the gate (output is for logging only).
            init = float(cfg.kalman.fusion_gate_init or -3.0)
            self.kalman_fusion_gate = nn.Parameter(torch.tensor(init))
            self.kalman_diagnostic_only = (cfg.kalman.mode == "diagnostic_probe")
        else:
            self.kalman = None
            self.kalman_fusion_gate = None
            self.kalman_diagnostic_only = False

        # ----- compute router -----------------------------------------
        if cfg.dram.router.enabled and cfg.kalman.enabled:
            self.router = ActiveInferenceRouter(
                d_model=cfg.d_model,
                k_max=cfg.dram.router.k_max,
                k_target=cfg.dram.router.k_target,
                init_kappa=cfg.dram.router.init_kappa,
            )
        else:
            self.router = EntropyRouter(
                d_model=cfg.d_model,
                k_max=cfg.dram.k_max,
                k_target=cfg.dram.k_iterations_default,
            )

        # ----- main backbone ------------------------------------------
        # Two layouts:
        #   "uniform_layers"          (legacy): every layer iterated K times
        #                              via TokenLevelDRAMBlock; per-token K*.
        #   "recurrent_middle_block"  (v3.1):  prefix(P) + tied-cell(C)·K +
        #                              suffix(S); per-block K* (one K per
        #                              batch element).
        self.cell_type = cfg.dram.cell_type
        if self.cell_type == "recurrent_middle_block":
            P = cfg.dram.prefix_layers
            C = cfg.dram.recurrent_cell_layers
            S = cfg.dram.suffix_layers
            assert C > 0, "recurrent_middle_block requires recurrent_cell_layers > 0"

            def _make_layer(idx: int) -> SPRLLayer:
                t = cfg.attention.layer_types[idx % len(cfg.attention.layer_types)]
                attn = make_attention(t, cfg.attention, cfg.d_model)
                ffn = make_ffn(cfg.moe, cfg.d_model)
                return SPRLLayer(cfg.d_model, attn_module=attn, ffn_module=ffn)

            prefix = [_make_layer(i) for i in range(P)]
            cell = [_make_layer(P + i) for i in range(C)]
            suffix = [_make_layer(P + C + i) for i in range(S)]
            self.middle = RecurrentMiddleBlock(prefix, cell, suffix)
            # Keep `self.layers` empty for the uniform path; downstream MoE
            # bookkeeping iterates over `self.iter_layers()` instead.
            self.layers = nn.ModuleList()
        else:
            # Legacy uniform-layer path.
            self.layers: List[DRAMBlock] = nn.ModuleList()
            for i in range(cfg.n_layers):
                layer_type = cfg.attention.layer_types[
                    i % len(cfg.attention.layer_types)
                ]
                attn = make_attention(layer_type, cfg.attention, cfg.d_model)
                ffn = make_ffn(cfg.moe, cfg.d_model)
                inner = SPRLLayer(cfg.d_model, attn_module=attn, ffn_module=ffn)
                block = TokenLevelDRAMBlock(
                    d_model=cfg.d_model,
                    attention_module=inner.attn,
                    ffn_module=inner.ffn,
                    use_depth_attention=cfg.dram.use_depth_attention,
                )
                block.norm1 = inner.norm_attn
                block.norm2 = inner.norm_ffn
                self.layers.append(block)
            self.middle = None

        # ----- Bet C: RG flow -----------------------------------------
        if cfg.dram.rg_flow.enabled:
            self.rg_flow = RGFlowRegularizer(
                d_model=cfg.d_model,
                rank=cfg.dram.rg_flow.rank_T_b,
                init_method=cfg.dram.rg_flow.init_method,
            )
        else:
            self.rg_flow = None

        # ----- output head --------------------------------------------
        self.byte_lm_head = ByteLMHead(
            d_model=cfg.d_model,
            byte_dim=cfg.patcher.byte_encoder_dim,
            n_layers=2,
            max_patch_bytes=cfg.patcher.max_patch_bytes,
            vocab=cfg.vocab_size,
        )

        # ----- MTP head ----------------------------------------------
        if cfg.mtp.enabled:
            self.mtp = MultiTokenPredictionHead(
                d_model=cfg.d_model,
                depth=cfg.mtp.depth,
                vocab=cfg.vocab_size,
            )
        else:
            self.mtp = None

    # ------------------------------------------------------------------
    def _iter_layers(self):
        """Iterate over all backbone layer modules irrespective of layout.

        For uniform layouts, yields each TokenLevelDRAMBlock. For middle-cell
        layouts, yields prefix + cell + suffix layers (the cell layers are
        yielded once even though they're applied K times — MoE accounting
        is on params, not iterations).
        """
        if self.middle is not None:
            for layer in self.middle.prefix:
                yield layer
            for layer in self.middle.cell:
                yield layer
            for layer in self.middle.suffix:
                yield layer
        else:
            for layer in self.layers:
                yield layer

    # ------------------------------------------------------------------
    def forward_from_patches(
        self,
        patch_emb: Tensor,
        future_targets: Optional[Tensor] = None,
        n_iter_default: int = 1,
    ) -> SPRLOutput:
        """Skip patcher; useful for unit tests and when you have pre-computed embeddings.

        patch_emb: [B, S_patches, d_model]
        future_targets: [B, S_patches, depth] for MTP loss (optional).
        """
        z = patch_emb
        diagnostics: dict = {}
        aux_losses: dict = {}
        all_iters: list[Tensor] = [z]

        # Pre-attention: optional Kalman observation.
        log_det_Lambda = None
        if self.kalman is not None:
            kal = self.kalman(z)
            mu = self.kalman.mean(kal["eta"], kal["D"], kal["U"])
            log_det_Lambda = kal["log_det_Lambda"]
            diagnostics["log_det_Lambda_mean"] = float(log_det_Lambda.mean().item())
            diagnostics["kalman_fusion_gate"] = float(
                torch.sigmoid(self.kalman_fusion_gate).item()
            )
            # Gated residual fusion. In `diagnostic_probe` mode the gate is
            # zeroed (Kalman output kept off the hidden stream).
            if not self.kalman_diagnostic_only:
                gate = torch.sigmoid(self.kalman_fusion_gate)
                z = z + gate * mu

        # Compute iteration plan. Per-token soft K* from the router; for
        # middle-cell layouts we aggregate to per-block (one K per batch elem).
        if isinstance(self.router, ActiveInferenceRouter):
            G = self.router.compute_G(z, log_det_Lambda)
        else:
            G = self.router.compute_G(z)
        k_per_token = self.router.k_star(G)  # [B, S], soft float (with STE)

        if self.cfg.dram.routing_granularity == "per_block":
            # Aggregate to one K per batch element. Mean is consistent with
            # the spec's per-block routing (§4.5.6).
            k_per_block_soft = k_per_token.float().mean(dim=-1)  # [B]
            diagnostics["mean_k"] = float(k_per_block_soft.mean().item())
            diagnostics["max_k"] = float(k_per_block_soft.max().item())
            k_block_int = (
                k_per_block_soft.detach()
                .round()
                .clamp(min=1.0, max=float(self.cfg.dram.k_max))
                .long()
            )
            if n_iter_default is not None:
                k_block_int = k_block_int.clamp(min=int(n_iter_default))
        else:
            k_block_int = None
            diagnostics["mean_k"] = float(k_per_token.mean().item())
            diagnostics["max_k"] = float(k_per_token.max().item())

        if self.cell_type == "recurrent_middle_block":
            assert k_block_int is not None, (
                "recurrent_middle_block requires per_block routing"
            )
            z, iter_outs = self.middle(
                z,
                k_per_block=k_block_int,
                return_iterations=(self.rg_flow is not None),
            )
            if self.rg_flow is not None and iter_outs:
                all_iters.extend(iter_outs[1:])
        else:
            # Legacy uniform path: per-token K* through TokenLevelDRAMBlock.
            k_int = (
                k_per_token.detach()
                .round()
                .clamp(min=1.0, max=float(self.cfg.dram.k_max))
                .long()
            )
            if n_iter_default is not None:
                k_int = k_int.clamp(min=int(n_iter_default))
            for layer in self.layers:
                z, hist = layer.forward_token_level(
                    z,
                    k_per_token=k_int,
                    record_history=(self.rg_flow is not None),
                    max_iter_cap=self.cfg.dram.k_max,
                )
                if self.rg_flow is not None:
                    all_iters.extend(hist[1:])

        if self.rg_flow is not None and len(all_iters) >= 2:
            aux_losses["rg_flow"] = self.rg_flow.loss(all_iters)
            diagnostics["T_b_norm"] = self.rg_flow.distance_from_identity()

        # MoE auxiliary losses (gather from every layer in either path).
        moe_aux = []
        for layer in self._iter_layers():
            ffn = getattr(layer, "ffn", None)
            if hasattr(ffn, "aux_loss"):
                a = ffn.aux_loss()
                if a is not None:
                    moe_aux.append(a)
        if moe_aux:
            aux_losses["moe_aux"] = torch.stack(moe_aux).sum()

        # Compute-budget Lagrangian (only meaningful with the active-inference router).
        if isinstance(self.router, ActiveInferenceRouter):
            aux_losses["compute_budget"] = self.router.compute_budget_loss(k_per_token.float())

        # Output: per-patch byte logits.
        # Treat each patch position as its own item for the head (no cross-patch context).
        B, S, d = z.shape
        z_flat = z.reshape(B * S, d)
        byte_logits = self.byte_lm_head(
            z_flat, n_bytes=self.cfg.patcher.max_patch_bytes
        )
        byte_logits = byte_logits.reshape(B, S, self.cfg.patcher.max_patch_bytes, -1)

        mtp_logits = None
        if self.mtp is not None and future_targets is not None:
            mtp_logits = self.mtp(z, future_targets)

        return SPRLOutput(
            byte_logits=byte_logits,
            mtp_logits=mtp_logits,
            aux_losses=aux_losses,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
