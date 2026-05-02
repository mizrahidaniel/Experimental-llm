"""RG-flow regularizer (Bet C): grad on T_b, ‖T_b − I‖ tracking."""

import torch

from sprl.recurrent.rg_flow import RGFlowRegularizer, marchenko_pastur_init


def test_marchenko_pastur_init_shape():
    U, V = marchenko_pastur_init(d=64, r=8)
    assert U.shape == (64, 8)
    assert V.shape == (64, 8)


def test_rg_loss_finite_and_grad():
    rg = RGFlowRegularizer(d_model=16, rank=4)
    z_iters = [torch.randn(2, 8, 16) for _ in range(4)]
    loss = rg.loss(z_iters)
    assert torch.isfinite(loss)
    loss.backward()
    assert rg.U.grad is not None and torch.isfinite(rg.U.grad).all()


def test_rg_loss_zero_when_history_too_short():
    rg = RGFlowRegularizer(d_model=8, rank=2)
    out = rg.loss([torch.randn(1, 4, 8)])
    assert out.item() == 0.0


def test_distance_from_identity_at_init():
    """At MP init, ‖T_b − I‖_F should be sizable, not near zero."""
    rg = RGFlowRegularizer(d_model=64, rank=8)
    dist = rg.distance_from_identity()
    # T_b = U V^T + s*I, with s≈0.05 and U,V random Gaussian: ‖T_b − I‖ ≈ ‖0.05·I − I‖ + small.
    assert dist > 0.5


def test_rg_step_residual_zero_when_perfectly_predicted():
    """If z_next = T_b z_prev exactly, the residual should be 0."""
    rg = RGFlowRegularizer(d_model=8, rank=2)
    z_prev = torch.randn(2, 4, 8)
    z_next = z_prev @ rg.matrix().t()
    res = rg.step_residual(z_prev, z_next)
    assert res.item() < 1e-10
