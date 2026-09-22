import math

import pytest
import torch
from torch import nn

from atariagent.training.learner import Trainer, _LearnerUnroll


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_regularization_weight(weight):
    with pytest.raises(ValueError, match="behavior_regularization_weight"):
        Trainer(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            consistency_network=nn.Identity(),
            behavior_regularization_weight=weight,
        )


def test_hydra_regularization_override():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from atariagent.training.config import register_train_agent_config

    register_train_agent_config()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"),
    ):
        default = compose(config_name="train_agent")
        enabled = compose(
            config_name="train_agent",
            overrides=["loss.behavior_regularization_weight=0.3"],
        )
    assert default.loss.behavior_regularization_weight == 0.0
    assert enabled.loss.behavior_regularization_weight == 0.3


def test_behavior_filter_and_stop_gradient():
    logits = torch.zeros(5, 2, requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, -1.0, 2.0, 0.0], requires_grad=True)
    values = torch.zeros(5, requires_grad=True)
    next_values = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0], requires_grad=True)
    loss = _LearnerUnroll._behavior_loss(
        logits,
        torch.tensor([[1], [0], [0], [-1], [0]]),
        rewards,
        values,
        next_values,
        torch.tensor([True, True, True, False, True]),
        0.5,
    )
    torch.testing.assert_close(loss, torch.tensor([math.log(2), 0, 0, 0, math.log(2)]))
    loss.sum().backward()
    assert rewards.grad is values.grad is next_values.grad is None
    torch.testing.assert_close(logits.grad[1:4], torch.zeros(3, 2))
    assert logits.grad[0, 1] < 0
    assert logits.grad[4, 0] < 0


class Representation(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, observations):
        return observations[:, :1, 0, 0] * 0 + self.weight


class Dynamics(nn.Module):
    def forward(self, state, action, hidden):
        # Uniform logits on [0, 1] give a positive, constant prefix.
        return state, None, state.expand(-1, 2) * 0


class Prediction(nn.Module):
    def forward(self, state):
        return state.expand(-1, 2), state.expand(-1, 2) * 0


class Consistency(nn.Module):
    def forward(self, state, target):
        return state, target


@pytest.mark.parametrize("horizon,expected_steps", [(1, 2), (2, 1)])
@pytest.mark.parametrize("weight", [0.0, 0.7])
def test_unroll_prefix_resets_weighting_padding_and_disabled(
    horizon, expected_steps, weight
):
    unroll = _LearnerUnroll(
        Representation(),
        Dynamics(),
        Prediction(),
        Consistency(),
        None,
        observation_dtype=torch.float32,
        unroll_steps=2,
        lstm_horizon=horizon,
        policy_weight=0.0,
        value_weight=0.0,
        reward_weight=0.0,
        consistency_weight=0.0,
        behavior_regularization_weight=weight,
        discount=0.9,
        support_min=0,
        support_max=1,
        priority_epsilon=1e-6,
    )
    outputs = unroll(
        torch.zeros(2, 3, 1, 1, 1, dtype=torch.uint8),
        torch.zeros(2, 2, 1, dtype=torch.long),
        torch.zeros(2, 2),
        torch.full((2, 3, 2), 0.5),
        torch.zeros(2, 3),
        torch.tensor([[True, True], [False, False]]),
        torch.ones(2, 3, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([0.4, 1.0]),
    )
    # Only the first sample is valid; importance weighting and 1/unroll_steps
    # apply to the objective, but not to the logged valid-action mean.
    assert outputs[0].item() == pytest.approx(
        weight * expected_steps * math.log(2) * 0.4 / 4
    )
    assert outputs[5][4].item() == pytest.approx(
        expected_steps * math.log(2) / 2 if weight else 0.0
    )
    outputs[0].backward()
