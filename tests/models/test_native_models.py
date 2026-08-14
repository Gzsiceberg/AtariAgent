from pathlib import Path

import torch

import atariagent.models as python_models
from atariagent.models.native import load_inference_checkpoint

CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "Alien-v5"
    / "muzero_latest.pt"
)


def test_cpp_models_load_real_training_checkpoint() -> None:
    models, checkpoint = load_inference_checkpoint(CHECKPOINT)

    assert set(models.representation.state_dict()) == set(checkpoint["representation"])
    assert set(models.dynamics.state_dict()) == set(checkpoint["dynamics"])
    assert set(models.prediction.state_dict()) == set(checkpoint["prediction"])


def test_cpp_checkpoint_inference_matches_python_models() -> None:
    native, checkpoint = load_inference_checkpoint(CHECKPOINT)
    python_representation = python_models.RepresentationNetwork(12).eval()
    python_dynamics = python_models.DynamicsNetwork(18).eval()
    python_prediction = python_models.PredictionNetwork(18).eval()
    python_representation.load_state_dict(checkpoint["representation"])
    python_dynamics.load_state_dict(checkpoint["dynamics"])
    python_prediction.load_state_dict(checkpoint["prediction"])

    generator = torch.Generator().manual_seed(42)
    observation = torch.randn(2, 12, 96, 96, generator=generator)
    action = torch.tensor([[0], [17]])

    with torch.inference_mode():
        python_state = python_representation(observation)
        native_state = native.representation(observation)
        python_next_state, python_hidden, python_prefix = python_dynamics(
            python_state, action
        )
        native_next_state, native_hidden, native_prefix = native.dynamics(
            native_state, action
        )
        python_second_state, python_second_hidden, python_second_prefix = (
            python_dynamics(python_next_state, action, python_hidden)
        )
        native_second_state, native_second_hidden, native_second_prefix = (
            native.dynamics(native_next_state, action, native_hidden)
        )
        python_policy, python_value = python_prediction(python_state)
        native_policy, native_value = native.prediction(native_state)

    torch.testing.assert_close(native_state, python_state)
    torch.testing.assert_close(native_next_state, python_next_state)
    torch.testing.assert_close(native_hidden[0], python_hidden[0])
    torch.testing.assert_close(native_hidden[1], python_hidden[1])
    torch.testing.assert_close(native_prefix, python_prefix)
    torch.testing.assert_close(native_second_state, python_second_state)
    torch.testing.assert_close(native_second_hidden[0], python_second_hidden[0])
    torch.testing.assert_close(native_second_hidden[1], python_second_hidden[1])
    torch.testing.assert_close(native_second_prefix, python_second_prefix)
    torch.testing.assert_close(native_policy, python_policy)
    torch.testing.assert_close(native_value, python_value)
