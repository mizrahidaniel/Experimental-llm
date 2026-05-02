"""Byte-level data utilities + PatcherCollator shape/correctness tests."""

from __future__ import annotations

import torch

from sprl.config import default_pilot_200m
from sprl.data.bytes import (
    IGNORE_INDEX,
    PAD_BYTE,
    CollatorConfig,
    PatcherCollator,
    chunk_byte_stream,
    derive_mtp_targets,
    plans_to_padded_tensors,
)
from sprl.patcher import ByteEncoder, ByteLM, EntropyPatcher


def test_chunk_byte_stream_reflows():
    pieces = [b"abcd", b"efgh", b"ijklmnop"]
    chunks = list(chunk_byte_stream(pieces, chunk_bytes=4, drop_last=True))
    assert len(chunks) == 4  # 16 bytes / 4
    assert chunks[0].tolist() == [ord("a"), ord("b"), ord("c"), ord("d")]
    assert chunks[3].tolist() == [ord("m"), ord("n"), ord("o"), ord("p")]
    for c in chunks:
        assert c.dtype == torch.long
        assert c.shape == (4,)


def test_chunk_byte_stream_drop_vs_pad():
    pieces = [b"abcde"]
    drop = list(chunk_byte_stream(pieces, chunk_bytes=4, drop_last=True))
    pad = list(chunk_byte_stream(pieces, chunk_bytes=4, drop_last=False))
    assert len(drop) == 1
    assert len(pad) == 2
    # The padded tail should contain PAD_BYTE.
    assert pad[1][-1].item() == PAD_BYTE


def test_chunk_byte_stream_string_inputs():
    chunks = list(chunk_byte_stream(["abcd"], chunk_bytes=4, drop_last=True))
    assert len(chunks) == 1
    assert chunks[0].tolist() == [ord("a"), ord("b"), ord("c"), ord("d")]


def test_plans_to_padded_tensors_basic():
    lm = ByteLM(dim=32, n_layers=1)
    p = EntropyPatcher(lm, threshold=100.0, min_patch_bytes=1, max_patch_bytes=4)
    bytes_seq = torch.arange(0, 16, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    plans = p.plan(bytes_seq)
    pb, ln, tgt, am = plans_to_padded_tensors(
        bytes_seq, plans, max_patch_bytes=4, max_seq_patches=8
    )
    # With threshold=inf, every patch saturates to max_patch_bytes=4 -> 4 patches.
    assert pb.shape == (2, 8, 4)
    assert ln.shape == (2, 8)
    assert tgt.shape == (2, 8, 4)
    assert am.shape == (2, 8)
    # First 4 slots are valid, remainder padded.
    assert am[0, :4].tolist() == [1, 1, 1, 1]
    assert am[0, 4:].tolist() == [0, 0, 0, 0]
    # Padded slots use PAD_BYTE on patch_bytes and IGNORE_INDEX on targets.
    assert (pb[0, 4:] == PAD_BYTE).all()
    assert (tgt[0, 4:] == IGNORE_INDEX).all()


def test_derive_mtp_targets_alignment():
    # Build a tiny mock: 4 patches, first byte 10, 20, 30, 40.
    targets = torch.full((1, 4, 3), IGNORE_INDEX)
    targets[0, 0, 0] = 10
    targets[0, 1, 0] = 20
    targets[0, 2, 0] = 30
    targets[0, 3, 0] = 40
    am = torch.tensor([[1, 1, 1, 1]])
    out = derive_mtp_targets(targets, am, depth=2)
    # depth-0 = next patch's first byte; depth-1 = patch+2's first byte.
    assert out[0, 0, 0].item() == 20
    assert out[0, 1, 0].item() == 30
    assert out[0, 0, 1].item() == 30
    assert out[0, 1, 1].item() == 40
    # Last position with no future -> IGNORE_INDEX.
    assert out[0, 3, 0].item() == IGNORE_INDEX


def test_patcher_collator_shapes_and_determinism():
    cfg = default_pilot_200m()
    cfg.patcher.entropy_threshold = 100.0  # force max-len patches deterministically
    P = cfg.patcher.max_patch_bytes
    lm = ByteLM(dim=cfg.patcher.byte_lm_dim, n_layers=cfg.patcher.byte_lm_layers)
    patcher = EntropyPatcher(
        lm, threshold=cfg.patcher.entropy_threshold, max_patch_bytes=P
    )
    enc = ByteEncoder(
        byte_dim=cfg.patcher.byte_encoder_dim,
        patch_dim=cfg.d_model,
        n_layers=2,
        max_patch_bytes=P,
    )
    cc = CollatorConfig(chunk_bytes=128, max_seq_patches=16, max_patch_bytes=P, batch_size=2, mtp_depth=2)
    coll = PatcherCollator(patcher, enc, cc)

    chunks = [torch.randint(0, 256, (128,), generator=torch.Generator().manual_seed(s)) for s in range(2)]
    out1 = coll.collate(chunks)
    out2 = coll.collate(chunks)
    assert out1["patch_emb"].shape == (2, 16, cfg.d_model)
    assert out1["byte_targets"].shape == (2, 16, P)
    assert out1["mtp_targets"].shape == (2, 16, 2)
    assert out1["attention_mask"].shape == (2, 16)
    # Determinism: same inputs -> same outputs.
    assert torch.allclose(out1["patch_emb"], out2["patch_emb"])
    # Padded patches have zeroed embeddings.
    pad_mask = (out1["attention_mask"] == 0).unsqueeze(-1).float()
    pad_emb = out1["patch_emb"] * pad_mask
    assert pad_emb.abs().sum().item() == 0.0


def test_patcher_collator_stream_batches():
    cfg = default_pilot_200m()
    cfg.patcher.entropy_threshold = 100.0
    P = cfg.patcher.max_patch_bytes
    lm = ByteLM(dim=cfg.patcher.byte_lm_dim, n_layers=1)
    patcher = EntropyPatcher(lm, threshold=100.0, max_patch_bytes=P)
    enc = ByteEncoder(
        byte_dim=cfg.patcher.byte_encoder_dim,
        patch_dim=cfg.d_model,
        n_layers=2,
        max_patch_bytes=P,
    )
    cc = CollatorConfig(chunk_bytes=64, max_seq_patches=8, max_patch_bytes=P, batch_size=2, mtp_depth=2)
    coll = PatcherCollator(patcher, enc, cc)

    # Stream 3 batches' worth of bytes.
    src = [b"x" * 64 * 6]
    batches = list(coll.stream_batches(src))
    assert len(batches) == 3
    for b in batches:
        assert b["patch_emb"].shape == (2, 8, cfg.d_model)
