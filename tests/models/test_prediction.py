import pytest
import torch
from torch import nn

from atariagent.models import PolicyNetwork, PredictionNetwork, ValueNetwork


def test_prediction_output_shapes() -> None:
    model = PredictionNetwork(action_space_size=18)

    policy, value = model(torch.randn(2, 64, 6, 6))

    assert policy.shape == (2, 18)
    assert value.shape == (2, 601)


@pytest.mark.parametrize(
    ("model", "output_size"),
    [(PolicyNetwork(18), 18), (ValueNetwork(), 601)],
)
def test_prediction_head_architecture(model: nn.Module, output_size: int) -> None:
    convolution = model.features[0]
    first_linear = model.projection[0]
    output_linear = model.projection[-1]

    assert isinstance(convolution, nn.Conv2d)
    assert convolution.kernel_size == (1, 1)
    assert (convolution.in_channels, convolution.out_channels) == (64, 16)
    assert convolution.bias is not None
    assert isinstance(first_linear, nn.Linear)
    assert (first_linear.in_features, first_linear.out_features) == (
        16 * 6 * 6,
        32,
    )
    assert isinstance(model.projection[2], nn.ELU)
    assert isinstance(output_linear, nn.Linear)
    assert (output_linear.in_features, output_linear.out_features) == (
        32,
        output_size,
    )


def test_prediction_has_a_shared_residual_block() -> None:
    model = PredictionNetwork(action_space_size=18)

    assert model.residual.conv1.in_channels == 64
    assert model.residual.conv1.out_channels == 64
    assert model.residual.conv1.kernel_size == (3, 3)


def test_prediction_output_layers_are_zero_initialized() -> None:
    model = PredictionNetwork(action_space_size=18)

    for head in (model.policy, model.value):
        output_layer = head.projection[-1]
        assert isinstance(output_layer, nn.Linear)
        assert torch.count_nonzero(output_layer.weight) == 0
        assert torch.count_nonzero(output_layer.bias) == 0

    policy, value = model(torch.randn(2, 64, 6, 6))
    assert torch.count_nonzero(policy) == 0
    assert torch.count_nonzero(value) == 0


def test_prediction_supports_custom_output_dimensions() -> None:
    model = PredictionNetwork(action_space_size=6, value_support_size=101)

    policy, value = model(torch.randn(2, 64, 6, 6))

    assert policy.shape == (2, 6)
    assert value.shape == (2, 101)


def test_prediction_supports_backward() -> None:
    model = PredictionNetwork(action_space_size=18)
    state = torch.randn(2, 64, 6, 6, requires_grad=True)

    policy, value = model(state)
    (policy.mean() + value.mean()).backward()

    assert state.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())
