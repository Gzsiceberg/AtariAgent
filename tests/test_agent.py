import random

import numpy as np
import torch
from torch import nn

from atariagent import (
    AgentOutput,
    AtariAgent,
    BatchedNetworkEvaluator,
    categorical_to_scalar,
)
from atariagent.search import SearchConfig, TreeSearch


class RecordingRepresentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.input_shapes: list[tuple[int, ...]] = []
        self.input_dtypes: list[torch.dtype] = []
        self.grad_modes: list[bool] = []
        self.training_modes: list[bool] = []

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(observations.shape[0])
        self.input_shapes.append(tuple(observations.shape))
        self.input_dtypes.append(observations.dtype)
        self.grad_modes.append(torch.is_grad_enabled())
        self.training_modes.append(self.training)
        return torch.zeros(observations.shape[0], 64, 6, 6, device=observations.device)


class RecordingDynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reward_prediction = nn.Identity()
        self.reward_prediction.hidden_size = 512
        self.batch_sizes: list[int] = []
        self.grad_modes: list[bool] = []
        self.training_modes: list[bool] = []

    def forward(self, states, actions, hidden):
        self.batch_sizes.append(states.shape[0])
        self.grad_modes.append(torch.is_grad_enabled())
        self.training_modes.append(self.training)
        next_hidden = (hidden[0] + 1, hidden[1] + 1)
        value_prefix = torch.zeros(states.shape[0], 601, device=states.device)
        return states, next_hidden, value_prefix


class RecordingPrediction(nn.Module):
    def __init__(self, action_space_size: int) -> None:
        super().__init__()
        self.action_space_size = action_space_size
        self.batch_sizes: list[int] = []
        self.grad_modes: list[bool] = []
        self.training_modes: list[bool] = []

    def forward(self, states):
        self.batch_sizes.append(states.shape[0])
        self.grad_modes.append(torch.is_grad_enabled())
        self.training_modes.append(self.training)
        policy = torch.zeros(
            states.shape[0], self.action_space_size, device=states.device
        )
        value = torch.zeros(states.shape[0], 601, device=states.device)
        return policy, value


def test_agent_batches_root_and_recurrent_network_inference() -> None:
    representation = RecordingRepresentation()
    dynamics = RecordingDynamics()
    prediction = RecordingPrediction(action_space_size=4)
    simulations = 4
    agent = AtariAgent(
        4,
        4,
        representation_network=representation,
        dynamics_network=dynamics,
        prediction_network=prediction,
        search_config=SearchConfig(num_simulations=simulations),
    )

    output = agent.act(torch.randn(3, 4, 96, 96, requires_grad=True))

    assert agent.mcts is agent.search
    assert isinstance(agent.recurrent_evaluator, BatchedNetworkEvaluator)
    assert not hasattr(agent, "_evaluate_recurrent_batch")
    assert isinstance(output, AgentOutput)
    assert len(output.actions) == 3
    assert all(action in range(4) for action in output.actions)
    assert len(output.search_results) == 3
    assert output.predicted_values == (0.0, 0.0, 0.0)
    assert all(
        np.isclose(result.target_policy.sum(), 1.0)
        for result in output.search_results
    )
    assert representation.batch_sizes == [3]
    assert dynamics.batch_sizes == [3] * simulations
    assert prediction.batch_sizes == [3] * (simulations + 1)


def test_packed_policy_search_returns_complete_results() -> None:
    dynamics = RecordingDynamics()
    prediction = RecordingPrediction(action_space_size=3)
    evaluator = BatchedNetworkEvaluator(
        dynamics,
        prediction,
        action_space_size=3,
        value_decoder=categorical_to_scalar,
        value_prefix_decoder=categorical_to_scalar,
    )
    states = torch.zeros(4, 64, 6, 6)
    policy_logits, value_logits = prediction(states)
    values = categorical_to_scalar(value_logits)
    simulations = 7

    mcts = TreeSearch(
        SearchConfig(num_simulations=simulations),
        evaluator=evaluator,
        rng=random.Random(4),
    )
    batch = mcts.search_batch(
        states,
        values,
        policy_logits,
        _deterministic_ties=True,
    )
    results = mcts.materialize_results(batch)

    assert batch.policy_targets.shape == (states.shape[0], 3)
    np.testing.assert_allclose(batch.policy_targets.sum(axis=1), 1.0)
    assert all(result.action in range(3) for result in results)


def test_agent_runs_every_network_without_gradients_and_in_eval_mode() -> None:
    representation = RecordingRepresentation()
    dynamics = RecordingDynamics()
    prediction = RecordingPrediction(action_space_size=2)
    agent = AtariAgent(
        4,
        2,
        representation_network=representation,
        dynamics_network=dynamics,
        prediction_network=prediction,
        search_config=SearchConfig(num_simulations=2),
    )
    assert agent.training

    agent.act(torch.randn(2, 4, 96, 96))

    modules = (representation, dynamics, prediction)
    assert all(not any(module.grad_modes) for module in modules)
    assert all(not any(module.training_modes) for module in modules)
    assert agent.training


def test_agent_prepares_raw_atari_observations() -> None:
    representation = RecordingRepresentation()
    agent = AtariAgent(
        12,
        3,
        representation_network=representation,
        dynamics_network=RecordingDynamics(),
        prediction_network=RecordingPrediction(3),
        search_config=SearchConfig(num_simulations=1),
    )
    observations = [
        np.full((4, 96, 96, 3), 255, dtype=np.uint8),
        np.zeros((4, 96, 96, 3), dtype=np.uint8),
    ]

    output = agent.act(observations)

    assert len(output.actions) == 2
    assert representation.input_shapes == [(2, 12, 96, 96)]
    assert representation.input_dtypes == [torch.float32]


def test_agent_accepts_one_unbatched_observation() -> None:
    agent = AtariAgent(
        4,
        3,
        representation_network=RecordingRepresentation(),
        dynamics_network=RecordingDynamics(),
        prediction_network=RecordingPrediction(3),
        search_config=SearchConfig(num_simulations=1),
    )

    output = agent.act(torch.randn(4, 96, 96))

    assert len(output.actions) == 1
    assert len(output.search_results) == 1


def test_categorical_to_scalar_uses_symmetric_efficientzero_support() -> None:
    center_logits = torch.full((1, 601), -1000.0)
    center_logits[0, 300] = 1000.0
    positive_logits = torch.full((1, 601), -1000.0)
    positive_logits[0, 301] = 1000.0

    center = categorical_to_scalar(center_logits)
    positive = categorical_to_scalar(positive_logits)

    torch.testing.assert_close(center, torch.zeros(1))
    assert positive.item() > 1.0
