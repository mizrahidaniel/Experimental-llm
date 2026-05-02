"""Disagreement scorer + diversity regularizer."""

import torch

from sprl.training.active_selection import (
    DisagreementSelector,
    batch_token_entropy,
    disagreement_score,
)


def test_disagreement_score_zero_when_distributions_match():
    logits = torch.randn(4, 8, 32)
    score = disagreement_score(logits, logits)
    assert score.shape == (4,)
    assert (score.abs() < 1e-5).all()


def test_disagreement_score_positive_when_distributions_differ():
    s = torch.randn(4, 8, 32)
    t = torch.randn(4, 8, 32) * 5.0  # very different scale ⇒ different distributions
    score = disagreement_score(s, t)
    assert (score > 0).all()


def test_selector_picks_top_k_by_score():
    """If we plant one obviously-disagreeing doc, the selector should pick it."""
    torch.manual_seed(0)
    pool = 16
    s = torch.randn(pool, 8, 32)
    t = s.clone()
    # Doc 7 is very different.
    t[7] = torch.randn(8, 32) * 10.0
    embeds = torch.randn(pool, 16)

    sel = DisagreementSelector(diversity_lambda=0.0)  # ignore diversity
    keep = sel.select(s, t, embeds, k=1)
    assert keep.tolist() == [7]


def test_diversity_penalty_pushes_against_repeats():
    """With a strong diversity penalty, the second-pick should not be a near-duplicate
    of the first pick even if it has higher disagreement."""
    pool = 4
    s = torch.zeros(pool, 4, 8)
    t = torch.randn(pool, 4, 8) * torch.tensor([5.0, 4.5, 1.0, 0.5]).reshape(-1, 1, 1)

    # Doc 0 and 1 have nearly-identical embeddings (forces diversity penalty).
    embeds = torch.tensor(
        [[1.0, 0.0], [0.999, 0.0], [0.0, 1.0], [0.0, 0.999]]
    ).expand(pool, 2).clone()
    embeds[2:] = torch.tensor([[0.0, 1.0], [0.0, 0.999]])

    sel = DisagreementSelector(diversity_lambda=10.0, history_size=10)
    first = sel.select(s, t, embeds, k=1)
    assert first.tolist() == [0]  # highest disagreement
    # Second pick: doc 1 has the second-highest disagreement, but it's near-dup
    # of doc 0 (which is in history). With λ=10 the penalty dominates ⇒ doc 2.
    second = sel.select(s, t, embeds, k=1)
    assert second.tolist() == [2]


def test_selector_history_evicts_old():
    sel = DisagreementSelector(diversity_lambda=0.0, history_size=3)
    embeds = torch.randn(5, 4)
    s = torch.randn(5, 2, 4)
    sel.select(s, s + torch.randn_like(s), embeds, k=5)
    # History capped at 3.
    assert len(sel._history) == 3


def test_batch_token_entropy_uniform_is_max():
    """A uniform batch of tokens should have entropy ≈ log(vocab)."""
    import math

    vocab = 8
    # Each id 0..7 appears once.
    tokens = torch.arange(vocab)
    H = batch_token_entropy(tokens, vocab)
    assert H == math.log(vocab) or abs(H - math.log(vocab)) < 1e-4


def test_batch_token_entropy_collapsed_is_zero():
    """All-same-token batch has entropy 0."""
    tokens = torch.zeros(100, dtype=torch.long)
    H = batch_token_entropy(tokens, vocab_size=256)
    assert H == 0.0


def test_selector_reset_clears_history():
    sel = DisagreementSelector(diversity_lambda=0.0)
    sel.select(
        torch.randn(2, 2, 4),
        torch.randn(2, 2, 4),
        torch.randn(2, 4),
        k=2,
    )
    assert sel._history
    sel.reset()
    assert not sel._history
