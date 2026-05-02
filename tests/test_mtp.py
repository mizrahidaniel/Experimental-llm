"""MTP head: forward shape and loss reduction on a tiny synthetic problem."""

import torch
import torch.nn.functional as F

from sprl.heads.mtp import MultiTokenPredictionHead


def test_mtp_forward_shape():
    head = MultiTokenPredictionHead(d_model=32, depth=2, vocab=64)
    h = torch.randn(2, 8, 32)
    targets = torch.randint(0, 64, (2, 8, 2))
    outs = head(h, targets)
    assert len(outs) == 2
    for o in outs:
        assert o.shape == (2, 8, 64)


def test_mtp_loss_decreases_with_training():
    head = MultiTokenPredictionHead(d_model=16, depth=2, vocab=8)
    h = torch.randn(4, 16, 16)
    targets = torch.randint(0, 8, (4, 16, 2))

    opt = torch.optim.AdamW(head.parameters(), lr=1.0e-2)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        outs = head(h, targets)
        loss = MultiTokenPredictionHead.loss(outs, targets, weight=1.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Loss should decrease materially.
    assert losses[-1] < losses[0] * 0.7, f"loss not decreasing: {losses[0]:.3f} → {losses[-1]:.3f}"
