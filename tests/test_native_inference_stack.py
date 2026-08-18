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
from atariagent.search import SearchConfig
from atariagent.search import TreeSearch as PythonTreeSearch

CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "Alien-v5"
)
CHECKPOINT = next(CHECKPOINT_DIRECTORY.glob("*_latest.pt"))


@pytest.fixture(scope="module")
def checkpoint_networks():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    representation_state = checkpoint["representation"]
    prediction_state = checkpoint["prediction"]
    in_channels = int(representation_state["stem.0.weight"].shape[1])
    action_space_size = int(prediction_state["policy.projection.3.weight"].shape[0])
    value_support_size = int(prediction_state["value.projection.3.weight"].shape[0])
    dynamics_state = checkpoint["dynamics"]
    if "action_projection.weight" not in dynamics_state:
        dynamics_state = python_models.DynamicsNetwork(action_space_size).state_dict()
    native_models = native.InferenceModels(
        representation=native.RepresentationNetwork(in_channels),
        dynamics=native.DynamicsNetwork(action_space_size),
        prediction=native.PredictionNetwork(action_space_size, value_support_size),
    )
    native_models.representation.load_state_dict(representation_state)
    native_models.dynamics.load_state_dict(dynamics_state)
    native_models.prediction.load_state_dict(prediction_state)
    native_models.eval()
    dynamics = python_models.DynamicsNetwork(action_space_size).eval()
    prediction = python_models.PredictionNetwork(
        action_space_size,
        value_support_size=value_support_size,
    ).eval()
    dynamics.load_state_dict(dynamics_state)
    prediction.load_state_dict(prediction_state)
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


@pytest.mark.parametrize("search_algorithm", ["puct", "gumbel"])
def test_native_complete_tree_search_matches_python(
    checkpoint_networks,
    search_algorithm: str,
) -> None:
    native_models, dynamics, prediction = checkpoint_networks
    config = SearchConfig(
        num_simulations=8,
        discount=0.9,
        value_prefix_horizon=3,
        search_algorithm=search_algorithm,
        num_top_actions=4,
    )
    python_evaluator = PythonBatchedNetworkEvaluator(
        dynamics,
        prediction,
        action_space_size=18,
        value_decoder=python_categorical_to_scalar,
        value_prefix_decoder=python_categorical_to_scalar,
    )
    python_search = PythonTreeSearch(
        config,
        evaluator=python_evaluator,
        rng=random.Random(0),
    )
    native_evaluator = native.BatchedNetworkEvaluator(
        native_models.dynamics,
        native_models.prediction,
        18,
    )
    native_search = native.TreeSearch(
        native_evaluator,
        config.num_simulations,
        config.discount,
        config.pb_c_init,
        config.pb_c_base,
        config.value_delta_max,
        config.dirichlet_alpha,
        config.root_exploration_fraction,
        config.value_prefix_horizon,
        0,
        config.search_algorithm,
        config.num_top_actions,
        config.c_visit,
        config.c_scale,
    )
    states = torch.randn(2, 64, 6, 6)
    with torch.inference_mode():
        policy_logits, value_logits = prediction(states)
        root_values = python_categorical_to_scalar(value_logits)
        expected = python_search.search_batch(
            states,
            root_values,
            policy_logits,
            _deterministic_ties=True,
        )
        search_output, root_values = native_search.search_batch(
            states,
            root_values,
            policy_logits,
            False,
            True,
        )

    if search_algorithm == "gumbel":
        np.testing.assert_allclose(
            search_output.numpy(), expected.policy_targets, rtol=1e-5, atol=1e-6
        )
    else:
        np.testing.assert_array_equal(search_output.numpy(), expected.visit_counts)
    np.testing.assert_allclose(
        root_values.numpy(), expected.root_values, rtol=1e-6, atol=1e-6
    )
