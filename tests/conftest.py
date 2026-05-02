import pytest
import torch

from sprl.utils.seeds import seed_everything


@pytest.fixture(autouse=True)
def _seed():
    seed_everything(0)
    yield


@pytest.fixture
def small_batch():
    return torch.randn(2, 16, 64)
