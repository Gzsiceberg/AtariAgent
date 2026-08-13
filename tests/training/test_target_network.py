from unittest.mock import patch

import pytest
import torch

from atariagent.models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig
from atariagent.training import ValueTargetNetwork


class _MeanRepresentation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mean = observations.flatten(1).mean(dim=1, keepdim=True)
        return mean * self.scale


class _BinaryPrediction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor):
        policy = torch.zeros(state.shape[0], 1, device=state.device)
        value = torch.cat((-state, state + self.bias), dim=1)
        return policy, value


class _PolicyPrediction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor):
        policy = torch.cat((state + self.bias, -state), dim=1)
        value = torch.zeros(state.shape[0], 2, device=state.device)
        return policy, value


class _PolicyDynamics(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.reward_prediction = torch.nn.LSTM(1, 1)

    def forward(self, state, action, hidden):
        del action, hidden
        batch_size = state.shape[0]
        next_state = state + self.bias
        next_hidden = (
            torch.zeros(1, batch_size, 1, device=state.device),
            torch.zeros(1, batch_size, 1, device=state.device),
        )
        value_prefix = torch.zeros(batch_size, 2, device=state.device)
        return next_state, next_hidden, value_prefix


def _bootstrap_batch() -> ReplayBatch:
    return ReplayBatch(
        frames=torch.zeros(2, 2, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 1, 1, dtype=torch.long),
        rewards=torch.zeros(2, 1),
        policy_targets=torch.zeros(2, 2, 1),
        value_targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        policy_mask=torch.zeros(2, 2, dtype=torch.bool),
        value_mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
        value_bootstrap_frames=torch.tensor(
            [[[[[255]]], [[[0]]]], [[[[0]]], [[[255]]]]],
            dtype=torch.uint8,
        ),
        value_bootstrap_values=torch.zeros(2, 2),
        value_bootstrap_discounts=torch.tensor([[0.5, 0.0], [0.25, 0.5]]),
        value_bootstrap_mask=torch.tensor(
            [[True, False], [True, True]], dtype=torch.bool
        ),
    )


def test_target_network_reanalyzes_direct_values_without_gradients() -> None:
    target = ValueTargetNetwork(
        _MeanRepresentation(),
        _BinaryPrediction(),
        support_min=0,
        support_max=1,
    )
    batch = _bootstrap_batch()
    fresh = torch.stack(
        tuple(
            target.decoded_values(
                batch.normalized_value_bootstrap_observation(offset)
            )
            for offset in range(batch.unroll_steps + 1)
        ),
        dim=1,
    )

    reanalyzed = target.reanalyze_values(batch)
    expected = batch.with_reanalyzed_value_targets(fresh)

    torch.testing.assert_close(reanalyzed.value_targets, expected.value_targets)
    assert reanalyzed.policy_targets is batch.policy_targets
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())


def test_target_network_reanalyzes_all_policy_samples() -> None:
    target = ValueTargetNetwork(
        _MeanRepresentation(),
        _PolicyPrediction(),
        dynamics=_PolicyDynamics(),
        action_space_size=2,
        mcts_config=MCTSConfig(num_simulations=1, value_prefix_horizon=1),
        rng_seed=7,
        support_min=0,
        support_max=1,
        chunk_size=16,
    )
    stored_policies = torch.zeros(100, 2, 2)
    stored_policies[:, 0] = 0.5
    batch = ReplayBatch(
        frames=torch.zeros(100, 2, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(100, 1, 1, dtype=torch.long),
        rewards=torch.zeros(100, 1),
        policy_targets=stored_policies,
        value_targets=torch.zeros(100, 2),
        action_mask=torch.ones(100, 1, dtype=torch.bool),
        policy_mask=torch.tensor([[True, False]]).expand(100, -1),
        value_mask=torch.ones(100, 2, dtype=torch.bool),
        indices=torch.arange(100),
        importance_weights=torch.ones(100),
    )
    original_policies = batch.policy_targets.clone()

    assert target._mcts is not None
    with patch.object(
        target._mcts,
        "search_batch",
        wraps=target._mcts.search_batch,
    ) as search_batch:
        reanalyzed = target.reanalyze_policies(batch)

    assert search_batch.call_count == 7
    assert all(
        call.kwargs["add_exploration_noise"]
        for call in search_batch.call_args_list
    )
    changed_samples = reanalyzed.policy_targets[:, 0].ne(0.5).any(dim=1)
    assert changed_samples.all()
    torch.testing.assert_close(
        reanalyzed.policy_targets[changed_samples, 0].sum(dim=1),
        torch.ones(100),
    )
    torch.testing.assert_close(
        reanalyzed.policy_targets[:, 1],
        original_policies[:, 1],
    )
    torch.testing.assert_close(batch.policy_targets, original_policies)


def test_policy_reanalysis_runs_with_atari_networks() -> None:
    target = ValueTargetNetwork(
        RepresentationNetwork(4),
        PredictionNetwork(action_space_size=2),
        dynamics=DynamicsNetwork(action_space_size=2),
        action_space_size=2,
        mcts_config=MCTSConfig(num_simulations=1),
        chunk_size=4,
    )
    batch = ReplayBatch(
        frames=torch.randint(0, 256, (2, 5, 1, 96, 96), dtype=torch.uint8),
        actions=torch.zeros(2, 1, 1, dtype=torch.long),
        rewards=torch.zeros(2, 1),
        policy_targets=torch.full((2, 2, 2), 0.5),
        value_targets=torch.zeros(2, 2),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        policy_mask=torch.ones(2, 2, dtype=torch.bool),
        value_mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
    )

    reanalyzed = target.reanalyze_policies(batch)

    torch.testing.assert_close(
        reanalyzed.policy_targets.sum(dim=-1),
        torch.ones(2, 2),
    )
    assert reanalyzed.policy_targets.ne(0.5).any()


def test_target_network_validates_constructor_chunk_size() -> None:
    with pytest.raises(TypeError, match="chunk_size must be an integer"):
        ValueTargetNetwork(
            _MeanRepresentation(),
            _BinaryPrediction(),
            chunk_size=True,
        )
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ValueTargetNetwork(
            _MeanRepresentation(),
            _BinaryPrediction(),
            chunk_size=0,
        )


def test_target_network_hard_copies_online_weights_and_buffers() -> None:
    representation = _MeanRepresentation()
    prediction = _BinaryPrediction()
    dynamics = _PolicyDynamics()
    target = ValueTargetNetwork(
        representation,
        prediction,
        dynamics=dynamics,
        action_space_size=1,
        support_min=0,
        support_max=1,
    )

    representation.scale.data.fill_(3.0)
    prediction.bias.data.fill_(2.0)
    dynamics.bias.data.fill_(4.0)
    assert target.representation.scale.item() == 1.0
    assert target.prediction.bias.item() == 0.0
    assert target.dynamics is not None
    assert target.dynamics.bias.item() == 0.0

    target.synchronize(representation, prediction, dynamics)

    assert target.representation.scale.item() == 3.0
    assert target.prediction.bias.item() == 2.0
    assert target.dynamics.bias.item() == 4.0
