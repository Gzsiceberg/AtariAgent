import math

import torch
import torch.nn.functional as functional

from atariagent.models import ConsistencyNetwork, DynamicsNetwork, RepresentationNetwork
from atariagent.training import DynamicsTrainer, scalar_reward_loss


def test_scalar_reward_loss_matches_cross_entropy_for_support_atom() -> None:
    logits = torch.randn(2, 601)
    target = torch.tensor([-1.0, 1.0])

    loss = scalar_reward_loss(logits, target)
    expected = functional.cross_entropy(
        logits, torch.tensor([299, 301]), reduction="none"
    )

    torch.testing.assert_close(loss, expected)


def test_dynamics_train_step_updates_zero_initialized_reward_output() -> None:
    representation = RepresentationNetwork(4)
    dynamics = DynamicsNetwork(action_space_size=3)
    consistency = ConsistencyNetwork(
        projection_dim=32,
        projection_hidden_dim=64,
        prediction_hidden_dim=16,
    )
    trainer = DynamicsTrainer(
        representation,
        dynamics,
        consistency,
        unroll_steps=1,
        lstm_horizon=1,
    )
    output_layer = dynamics.reward_prediction.projection[-1]
    initial_weight = output_layer.weight.detach().clone()

    metrics = trainer.train_step(
        torch.randn(2, 2, 4, 96, 96),
        torch.tensor([[[0]], [[2]]]),
        torch.tensor([[0.0], [1.0]]),
    )

    assert metrics.loss > 0
    assert metrics.reward_loss > 0
    assert math.isfinite(metrics.consistency_loss)
    assert not torch.equal(output_layer.weight, initial_weight)
