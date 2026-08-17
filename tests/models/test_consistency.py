import torch

from atariagent.models import ConsistencyNetwork, consist_loss_func


def test_consistency_network_uses_efficientzero_v1_predictor_width() -> None:
    model = ConsistencyNetwork()

    assert model.predictor.network[0].out_features == 512


def test_consistency_network_shapes_and_stop_gradient() -> None:
    model = ConsistencyNetwork(
        projection_dim=32,
        projection_hidden_dim=64,
        prediction_hidden_dim=16,
    )
    predicted_state = torch.randn(2, 64, 6, 6, requires_grad=True)
    target_state = torch.randn(2, 64, 6, 6, requires_grad=True)

    prediction, target = model(predicted_state, target_state)
    loss = consist_loss_func(prediction, target).mean()
    loss.backward()

    assert prediction.shape == (2, 32)
    assert target.shape == (2, 32)
    assert predicted_state.grad is not None
    assert target_state.grad is None


def test_consistency_loss_is_negative_one_for_identical_projections() -> None:
    projection = torch.randn(2, 32)

    loss = consist_loss_func(projection, projection)

    torch.testing.assert_close(loss, -torch.ones(2), atol=1e-6, rtol=0)
