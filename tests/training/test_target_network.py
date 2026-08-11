import torch

from atariagent.replay import ReplayBatch
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


def _bootstrap_batch() -> ReplayBatch:
    return ReplayBatch(
        frames=torch.zeros(2, 2, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 1, 1, dtype=torch.long),
        rewards=torch.zeros(2, 1),
        policy_targets=torch.zeros(2, 2, 1),
        root_values=torch.zeros(2, 2),
        value_targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        target_mask=torch.ones(2, 2, dtype=torch.bool),
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

    reanalyzed = target.reanalyze(batch)
    expected = batch.with_reanalyzed_value_targets(fresh)

    torch.testing.assert_close(reanalyzed.value_targets, expected.value_targets)
    assert reanalyzed.policy_targets is batch.policy_targets
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())


def test_target_network_hard_copies_online_weights_and_buffers() -> None:
    representation = _MeanRepresentation()
    prediction = _BinaryPrediction()
    target = ValueTargetNetwork(
        representation,
        prediction,
        support_min=0,
        support_max=1,
    )

    representation.scale.data.fill_(3.0)
    prediction.bias.data.fill_(2.0)
    assert target.representation.scale.item() == 1.0
    assert target.prediction.bias.item() == 0.0

    target.synchronize(representation, prediction)

    assert target.representation.scale.item() == 3.0
    assert target.prediction.bias.item() == 2.0
