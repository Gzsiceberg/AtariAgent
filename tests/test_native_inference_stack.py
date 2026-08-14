import random
from pathlib import Path

import numpy as np
import pytest
import torch

import atariagent.models as python_models
from atariagent.agent import (
    BatchedNetworkEvaluator as PythonBatchedNetworkEvaluator,
)
from atariagent.agent import categorical_to_scalar as python_categorical_to_scalar
from atariagent.models import native
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTS as PythonMCTS
from atariagent.search import MCTSConfig
from atariagent.training import ValueTargetNetwork as PythonValueTargetNetwork

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "Alien-v5"
    / "muzero_latest.pt"
)


@pytest.fixture(scope="module")
def checkpoint_networks():
    native_models, checkpoint = native.load_inference_checkpoint(CHECKPOINT)
    dynamics = python_models.DynamicsNetwork(18).eval()
    prediction = python_models.PredictionNetwork(18).eval()
    dynamics.load_state_dict(checkpoint["dynamics"])
    prediction.load_state_dict(checkpoint["prediction"])
    return native_models, dynamics, prediction


def test_native_batched_evaluator_matches_python(checkpoint_networks) -> None:
    native_models, dynamics, prediction = checkpoint_networks
    python_evaluator = PythonBatchedNetworkEvaluator(
        dynamics,
        prediction,
        action_space_size=18,
        value_decoder=python_categorical_to_scalar,
        value_prefix_decoder=python_categorical_to_scalar,
    )
    native_evaluator = native.BatchedNetworkEvaluator(
        native_models.dynamics,
        native_models.prediction,
        18,
    )
    states = torch.randn(3, 64, 6, 6)
    actions = torch.tensor([[0], [7], [17]])
    hidden = python_evaluator.initial_hidden(
        3,
        device=states.device,
        dtype=states.dtype,
    )
    reset = torch.tensor([False, True, False])

    with torch.inference_mode():
        expected = python_evaluator.evaluate_tensors(states, actions, hidden, reset)
        actual = native_evaluator.evaluate_tensors(states, actions, hidden, reset)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1][0], expected[1][0])
    torch.testing.assert_close(actual[1][1], expected[1][1])
    torch.testing.assert_close(actual[2], expected[2])
    torch.testing.assert_close(actual[3], expected[3])
    torch.testing.assert_close(actual[4], expected[4])


def test_native_complete_mcts_matches_python(checkpoint_networks) -> None:
    native_models, dynamics, prediction = checkpoint_networks
    config = MCTSConfig(
        num_simulations=8,
        discount=0.9,
        value_prefix_horizon=3,
    )
    python_evaluator = PythonBatchedNetworkEvaluator(
        dynamics,
        prediction,
        action_space_size=18,
        value_decoder=python_categorical_to_scalar,
        value_prefix_decoder=python_categorical_to_scalar,
    )
    python_mcts = PythonMCTS(
        config,
        evaluator=python_evaluator,
        rng=random.Random(0),
    )
    native_mcts = native.make_mcts(native_models, 18, config, seed=0)
    states = torch.randn(2, 64, 6, 6)
    with torch.inference_mode():
        policy_logits, value_logits = prediction(states)
        root_values = python_categorical_to_scalar(value_logits)
        expected = python_mcts.search_batch(
            states,
            root_values,
            policy_logits,
            _deterministic_ties=True,
        )
        visit_counts, root_values = native_mcts.search_batch(
            states,
            root_values,
            policy_logits,
            False,
            True,
        )

    np.testing.assert_array_equal(visit_counts.numpy(), expected.visit_counts)
    np.testing.assert_allclose(
        root_values.numpy(), expected.root_values, rtol=1e-6, atol=1e-6
    )


def test_native_value_target_matches_python_reanalysis() -> None:
    python_representation = python_models.RepresentationNetwork(12).eval()
    python_prediction = python_models.PredictionNetwork(1).eval()
    python_dynamics = python_models.DynamicsNetwork(1).eval()
    native_models = native.InferenceModels(
        representation=native.RepresentationNetwork(12),
        dynamics=native.DynamicsNetwork(1),
        prediction=native.PredictionNetwork(1),
    )
    native_models.representation.load_state_dict(python_representation.state_dict())
    native_models.dynamics.load_state_dict(python_dynamics.state_dict())
    native_models.prediction.load_state_dict(python_prediction.state_dict())
    native_models.eval()
    config = MCTSConfig(
        num_simulations=2,
        root_exploration_fraction=0.0,
    )
    python_target = PythonValueTargetNetwork(
        python_representation,
        python_prediction,
        dynamics=python_dynamics,
        action_space_size=1,
        mcts_config=config,
        rng_seed=0,
        chunk_size=4,
    )
    native_target = native.make_value_target(
        native_models,
        1,
        config,
        seed=0,
        chunk_size=4,
    )
    batch = ReplayBatch(
        frames=torch.randint(0, 256, (2, 5, 3, 96, 96), dtype=torch.uint8),
        actions=torch.zeros(2, 1, 1, dtype=torch.long),
        rewards=torch.zeros(2, 1),
        policy_targets=torch.zeros(2, 2, 1),
        value_targets=torch.randn(2, 2),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        policy_mask=torch.ones(2, 2, dtype=torch.bool),
        value_mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
        value_bootstrap_frames=torch.randint(
            0, 256, (2, 5, 3, 96, 96), dtype=torch.uint8
        ),
        value_bootstrap_values=torch.randn(2, 2),
        value_bootstrap_discounts=torch.rand(2, 2),
        value_bootstrap_mask=torch.ones(2, 2, dtype=torch.bool),
    )

    with torch.inference_mode():
        expected = python_target.reanalyze_batch(batch)
        actual = native_target.reanalyze_batch(batch, False, True)

    torch.testing.assert_close(actual.value_targets, expected.value_targets)
    torch.testing.assert_close(actual.policy_targets, expected.policy_targets)
