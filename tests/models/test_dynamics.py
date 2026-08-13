import torch
from torch import nn

from atariagent.models import DynamicsNetwork, RewardPredictionNetwork


def test_dynamics_output_shapes() -> None:
    model = DynamicsNetwork(action_space_size=18)
    state = torch.randn(2, 64, 6, 6)
    action = torch.tensor([[0], [17]])

    next_state, (hidden, cell), value_prefix = model(state, action)

    assert model.scale_state_gradient
    assert next_state.shape == (2, 64, 6, 6)
    assert hidden.shape == (1, 2, 512)
    assert cell.shape == (1, 2, 512)
    assert value_prefix.shape == (2, 601)


def test_dynamics_uses_one_action_plane_and_preserves_resolution() -> None:
    model = DynamicsNetwork(action_space_size=18)
    convolution = model.transition[0]

    assert isinstance(convolution, nn.Conv2d)
    assert convolution.in_channels == 65
    assert convolution.out_channels == 64
    assert convolution.kernel_size == (3, 3)
    assert convolution.stride == (1, 1)


def test_reward_prediction_architecture() -> None:
    model = RewardPredictionNetwork()

    convolution = model.features[0]
    first_linear = model.projection[0]
    output_linear = model.projection[-1]

    assert isinstance(convolution, nn.Conv2d)
    assert convolution.kernel_size == (1, 1)
    assert convolution.out_channels == 16
    assert convolution.bias is not None
    assert model.lstm.input_size == 16 * 6 * 6
    assert model.lstm.hidden_size == 512
    assert isinstance(first_linear, nn.Linear)
    assert (first_linear.in_features, first_linear.out_features) == (512, 32)
    assert isinstance(output_linear, nn.Linear)
    assert (output_linear.in_features, output_linear.out_features) == (32, 601)


def test_reward_output_layer_is_zero_initialized() -> None:
    model = DynamicsNetwork(action_space_size=18)
    output_layer = model.reward_prediction.projection[-1]

    assert isinstance(output_layer, nn.Linear)
    assert torch.count_nonzero(output_layer.weight) == 0
    assert torch.count_nonzero(output_layer.bias) == 0

    _, _, value_prefix = model(
        torch.randn(2, 64, 6, 6), torch.tensor([[1], [2]])
    )
    assert torch.count_nonzero(value_prefix) == 0


def test_reward_hidden_can_be_carried_across_recurrent_steps() -> None:
    model = DynamicsNetwork(action_space_size=18)
    state = torch.randn(2, 64, 6, 6)
    action = torch.tensor([[1], [2]])

    next_state, hidden, _ = model(state, action)
    _, next_hidden, _ = model(next_state, action, hidden)

    assert not torch.equal(next_hidden[0], hidden[0])
    assert not torch.equal(next_hidden[1], hidden[1])


def test_dynamics_supports_backward() -> None:
    model = DynamicsNetwork(action_space_size=18)
    state = torch.randn(2, 64, 6, 6, requires_grad=True)

    next_state, _, _ = model(state, torch.tensor([[1], [2]]))
    next_state.mean().backward()

    assert state.grad is not None
    transition_parameters = list(model.transition.parameters())
    assert all(parameter.grad is not None for parameter in transition_parameters)
