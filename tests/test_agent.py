import numpy as np
import torch
from torch import nn

from atariagent import AgentOutput, AtariAgent, categorical_to_scalar
from atariagent.search import MCTSConfig


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
        return torch.zeros(
            observations.shape[0], 64, 6, 6, device=observations.device
        )


class RecordingDynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        mcts_config=MCTSConfig(num_simulations=simulations),
    )

    output = agent(torch.randn(3, 4, 96, 96, requires_grad=True))

    assert isinstance(output, AgentOutput)
    assert len(output.actions) == 3
    assert all(action in range(4) for action in output.actions)
    assert len(output.search_results) == 3
    assert all(result.root.visit_count == simulations + 1 for result in output.search_results)
    assert representation.batch_sizes == [3]
    assert dynamics.batch_sizes == [3] * simulations
    assert prediction.batch_sizes == [3] * (simulations + 1)


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
        mcts_config=MCTSConfig(num_simulations=2),
    )
    assert agent.training

    agent(torch.randn(2, 4, 96, 96))

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
        mcts_config=MCTSConfig(num_simulations=1),
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
        mcts_config=MCTSConfig(num_simulations=1),
    )

    output = agent(torch.randn(4, 96, 96))

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
