"""Config validation: BLT byte vocab + teacher_token_kl is incoherent.

Direct KL between a 256-class byte distribution and a typical 32K-vocab
teacher BPE distribution is shape-mismatched at best, semantically
meaningless at worst. Validation must hard-error.
"""

import pytest

from sprl.config import SPRLConfig


def test_blt_plus_teacher_token_kl_errors():
    cfg = SPRLConfig()
    cfg.patcher.enabled = True
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "teacher_token_kl"
    with pytest.raises(ValueError, match="BLT byte/patch logits"):
        cfg.validate()


def test_blt_plus_sequence_level_distill_ok():
    cfg = SPRLConfig()
    cfg.patcher.enabled = True
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "sequence_level"
    cfg.validate()  # no raise


def test_blt_plus_distill_disabled_ok():
    cfg = SPRLConfig()
    cfg.patcher.enabled = True
    cfg.training.distill_enabled = False
    cfg.training.distill_mode = "teacher_token_kl"  # mode irrelevant when off
    cfg.validate()  # no raise


def test_no_blt_plus_teacher_token_kl_ok():
    cfg = SPRLConfig()
    cfg.patcher.enabled = False
    cfg.training.distill_enabled = True
    cfg.training.distill_mode = "teacher_token_kl"
    cfg.validate()


def test_active_selection_requires_distill():
    cfg = SPRLConfig()
    cfg.patcher.enabled = False
    cfg.training.active_selection_enabled = True
    cfg.training.distill_enabled = False
    with pytest.raises(ValueError, match="active_selection_enabled"):
        cfg.validate()


def test_iterative_redistill_requires_distill():
    cfg = SPRLConfig()
    cfg.patcher.enabled = False
    cfg.training.iterative_redistill_enabled = True
    cfg.training.distill_enabled = False
    with pytest.raises(ValueError, match="iterative_redistill_enabled"):
        cfg.validate()
