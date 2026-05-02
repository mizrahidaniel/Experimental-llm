# SPRL-v2 architecture

## Central equation

```
z_0     = Patch(b)                                     // BLT-style entropy patcher
(η,Λ)_t = KalmanScan(z_t, ...)                         // Bet A — info-form belief memory
z_{k+1} = T_θ( DSA-MLA + Tropical + DRAM(z_k) ) + z_k  // Bet B + DRAM core
k*(t)   = ActiveInferenceRouter(η, Λ, ...)             // Bet A's uncertainty + free-energy router
                                                       // (replaces PIG)
RG-reg  = ‖z_{k+1} − T_b z_k‖²                         // Bet C — RG-flow regularizer
output_t = Decode_byte(z_{k*(t)})                      // MTP + byte LM head
```

## Pipeline

1. **Bytes → patches.** A frozen ~10M-param byte LM (BLT) computes per-position
   entropy. Patches are closed at high-entropy positions or at `max_patch_bytes`,
   yielding ~6-byte average patches.
2. **Per-patch encoding.** A 4-layer byte encoder + cross-attention pool
   produces a single `d_model` embedding per patch.
3. **Belief memory (Bet A, optional).** An information-form Kalman filter with
   low-rank precision Λ = D + UUᵀ is run as an associative scan over patch
   positions. Output: `η_t`, `Λ_t`, `log det Λ_t`.
4. **Compute routing.** The active-inference router combines `-log det Λ_t`
   (epistemic value) with a tiny scout head's predicted next-byte loss
   (pragmatic value) to produce per-token iteration counts `K*(t) ∈ [1, K_max]`.
   When Bet A is off, an entropy-only router (MoR-style) is used instead.
5. **Backbone.** `n_layers` layers, alternating between:
   - **DSA-under-MLA** layers (full-context sparse attention): MLA latent
     compresses K/V; a lightning indexer picks top-k keys per query; HISA
     hierarchical block→token cascade reduces indexer cost to O(L^1.5).
   - **Sliding-window** layers (window=1024 patches).
6. **Depth recurrence (DRAM).** Each layer is a weight-tied block iterated
   `K*(t)` times. Each iteration includes:
   - Local attention (DSA or sliding).
   - Depth-attention over previous iteration states `{z_0, …, z_{k-1}}`.
   - SwiGLU FFN or fine-grained MoE.
7. **Bet B.** In MLA layers, 1 of every 4 heads runs in tropical (max-plus)
   algebra instead of softmax, with a softmax-with-large-β surrogate during
   the warmup phase to keep gradients smooth.
8. **Bet C.** The RG-flow regularizer adds `α · ‖z_{k+1} − T_b z_k‖²` per
   iteration, where `T_b = U V^T + s · I` (rank-32, Marchenko-Pastur init).
9. **MoE.** 64 routed + 4 shared SwiGLU experts (heterogeneous: 32 small,
   32 1.5×). Top-6 routing with auxiliary-loss-free balancing (DeepSeek-V3
   bias-update rule).
10. **MTP head.** Predicts next-2 tokens via a depth-2 auxiliary stack;
    contributes `0.3 · L_MTP` to the loss; enables 2× speculative decoding.
11. **Byte LM head.** Cross-attention from the per-patch latent back to bytes
    via a small ByteDecoder + tied byte vocabulary head.

## Component locations

| Component | File |
|---|---|
| BLT entropy patcher | `sprl/patcher/blt_patcher.py` |
| MLA | `sprl/attention/mla.py` |
| DSA + lightning indexer | `sprl/attention/dsa.py` |
| HISA hierarchical indexer | `sprl/attention/hisa.py` |
| Sliding window | `sprl/attention/sliding.py` |
| Tropical heads (Bet B) | `sprl/attention/tropical.py` |
| Kalman info memory (Bet A) | `sprl/memory/kalman_info.py` |
| Parallel scan | `sprl/memory/parallel_scan.py` |
| Low-rank precision (SMW) | `sprl/memory/low_rank_precision.py` |
| Active-inference router | `sprl/recurrent/active_inference_router.py` |
| DRAM block | `sprl/recurrent/dram_block.py` |
| Depth attention | `sprl/recurrent/depth_attention.py` |
| RG flow (Bet C) | `sprl/recurrent/rg_flow.py` |
| Fine-grained MoE | `sprl/experts/moe.py` |
| ALF balancer | `sprl/experts/alf_balancer.py` |
| MTP head | `sprl/heads/mtp.py` |
| Byte LM head | `sprl/heads/byte_lm_head.py` |
| Top-level model | `sprl/model.py` |

## Memory budget (target 200M pilot)

The pilot config (`configs/pilot_200m_bf16.yaml`) has 16 routed + 2 shared
experts at `expert_dim=1024`, `d_model=768`, `n_layers=12`. The total
parameter count is ~700M (with the heterogeneous MoE multiplier); active
params per token after top-2 routing are ≈ 150M.

For the full 1.5–2B-active config (Stage 3), see the spec's table; expect
~16 GB tight on a 5080 with NVFP4, gradient checkpointing, and 8-bit AdamW.
