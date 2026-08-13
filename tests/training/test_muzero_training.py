from copy import deepcopy

import pytest
import torch
import torch.nn.functional as functional

from atariagent.models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from atariagent.replay_batch import ReplayBatch
from atariagent.training import MuZeroTrainer


def test_scalar_categorical_loss_interpolates_transformed_target() -> None:
    logits = torch.randn(2, 601)
    targets = torch.tensor([0.0, 1.0])
    batch = ReplayBatch(
        frames=torch.zeros(2, 2, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 1, 1, dtype=torch.long),
        rewards=torch.zeros(2, 1),
        policy_targets=torch.zeros(2, 2, 1),
        value_targets=torch.stack((targets, torch.zeros_like(targets)), dim=1),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        policy_mask=torch.zeros(2, 2, dtype=torch.bool),
        value_mask=torch.tensor([[True, False], [True, False]]),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
    )

    _, loss = batch.prediction_losses(
        torch.zeros(2, 1), logits, offset=0
    )
    transformed = (
        targets.sign() * (torch.sqrt(targets.abs() + 1.0) - 1.0)
        + 0.001 * targets
        + 300
    )
    lower = transformed.floor().long()
    upper = transformed.ceil().long()
    upper_weight = transformed - lower
    expected = (
        (1.0 - upper_weight)
        * functional.cross_entropy(logits, lower, reduction="none")
        + upper_weight
        * functional.cross_entropy(logits, upper, reduction="none")
    )

    torch.testing.assert_close(loss, expected)


def test_optimized_scalar_loss_matches_double_cross_entropy_and_gradients() -> None:
    logits = torch.randn(7, 601, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    targets = torch.tensor([-1e9, -2.5, 0.0, 1.0, 3.25, 300.0, 1e9])

    actual = ReplayBatch._scalar_loss(
        logits,
        targets,
        support_min=-300,
        support_max=300,
    )
    transformed = (
        targets.sign() * (torch.sqrt(targets.abs() + 1.0) - 1.0)
        + 0.001 * targets
    ).clamp(-300, 300) + 300
    lower = transformed.floor().long()
    upper = transformed.ceil().long()
    upper_weight = transformed - lower
    expected = (
        (1.0 - upper_weight)
        * functional.cross_entropy(reference_logits, lower, reduction="none")
        + upper_weight
        * functional.cross_entropy(reference_logits, upper, reduction="none")
    )

    actual_gradient = torch.autograd.grad(actual.sum(), logits)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), reference_logits)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_optimized_scalar_loss_is_safe_under_bfloat16_autocast() -> None:
    logits = torch.randn(4, 601, requires_grad=True)
    targets = torch.tensor([-300.0, 0.0, 1.5, 300.0])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = ReplayBatch._scalar_loss(
            logits,
            targets,
            support_min=-300,
            support_max=300,
        )
    loss.mean().backward()

    assert loss.dtype == torch.float32
    assert torch.all(torch.isfinite(loss))
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_value_prefix_targets_reset_at_lstm_horizon_and_respect_mask() -> None:
    rewards = torch.tensor([[1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0]])
    mask = torch.tensor(
        [[True, True, True, True], [True, False, False, False]]
    )

    batch = ReplayBatch(
        frames=torch.zeros(2, 5, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 4, 1, dtype=torch.long),
        rewards=rewards,
        policy_targets=torch.zeros(2, 5, 1),
        value_targets=torch.zeros(2, 5),
        action_mask=mask,
        policy_mask=torch.zeros(2, 5, dtype=torch.bool),
        value_mask=torch.zeros(2, 5, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
    )

    torch.testing.assert_close(
        batch.value_prefix_targets(lstm_horizon=2),
        torch.tensor([[1.0, 3.0, 4.0, 12.0], [1.0, 1.0, 0.0, 0.0]]),
    )


def test_muzero_train_step_updates_all_supervised_output_heads() -> None:
    representation = RepresentationNetwork(4)
    dynamics = DynamicsNetwork(action_space_size=3)
    prediction = PredictionNetwork(action_space_size=3)
    trainer = MuZeroTrainer(
        representation,
        dynamics,
        prediction,
        lr_warmup_steps=0,
        unroll_steps=1,
        lstm_horizon=1,
    )
    policy_output = prediction.policy.projection[-1]
    value_output = prediction.value.projection[-1]
    reward_output = dynamics.reward_prediction.projection[-1]
    assert not hasattr(trainer, "target_network")
    initial_policy = policy_output.weight.detach().clone()
    initial_value = value_output.weight.detach().clone()
    initial_reward = reward_output.weight.detach().clone()

    batch = ReplayBatch(
        frames=torch.randint(0, 256, (2, 5, 1, 96, 96), dtype=torch.uint8),
        actions=torch.tensor([[[0]], [[2]]]),
        rewards=torch.tensor([[1.0], [-1.0]]),
        policy_targets=torch.tensor(
            [
                [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
                [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
            ]
        ),
        value_targets=torch.tensor([[1.0, 0.5], [-1.0, -0.5]]),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        policy_mask=torch.ones(2, 2, dtype=torch.bool),
        value_mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
        value_bootstrap_frames=torch.randint(
            0, 256, (2, 5, 1, 96, 96), dtype=torch.uint8
        ),
        value_bootstrap_values=torch.zeros(2, 2),
        value_bootstrap_discounts=torch.ones(2, 2),
        value_bootstrap_mask=torch.ones(2, 2, dtype=torch.bool),
    )

    metrics = trainer.train_step(batch)

    assert metrics.loss == pytest.approx(
        metrics.policy_loss
        + 0.25 * metrics.value_loss
        + metrics.reward_loss
    )
    assert metrics.policy_loss > 0.0
    assert metrics.value_loss > 0.0
    assert metrics.reward_loss > 0.0
    assert metrics.learning_rate == pytest.approx(0.2)
    assert metrics.priorities == pytest.approx((1.000001, 1.000001))
    assert not torch.equal(policy_output.weight, initial_policy)
    assert not torch.equal(value_output.weight, initial_value)
    assert not torch.equal(reward_output.weight, initial_reward)


class _ScalarRepresentation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.weight.expand(observations.shape[0], 1)


class _IdentityDynamics(torch.nn.Module):
    def __init__(self, *, scale_state_gradient=True):
        super().__init__()
        self.scale_state_gradient = scale_state_gradient

    def forward(self, state, action, hidden):
        next_state = state + 0.0
        if self.scale_state_gradient:
            next_state = next_state * 0.5 + next_state.detach() * 0.5
        value_prefix = torch.cat((next_state * 0.0, next_state * 0.0), dim=1)
        return next_state, None, value_prefix


class _ScalarPrediction(torch.nn.Module):
    def forward(self, state):
        policy = torch.cat((state, torch.zeros_like(state)), dim=1)
        value = torch.cat((state * 0.0, state * 0.0), dim=1)
        return policy, value


class _RewardIdentityDynamics(_IdentityDynamics):
    def forward(self, state, action, hidden):
        next_state, _, _ = super().forward(state, action, hidden)
        value_prefix = torch.cat((next_state, torch.zeros_like(next_state)), dim=1)
        return next_state, None, value_prefix


def test_complete_compiled_unroll_matches_eager_update(monkeypatch) -> None:
    compile_arguments = {}

    def identity_compile(module, **kwargs):
        compile_arguments.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", identity_compile)
    representation = _ScalarRepresentation()
    dynamics = _RewardIdentityDynamics()
    prediction = _ScalarPrediction()
    compiled_modules = tuple(
        deepcopy(module) for module in (representation, dynamics, prediction)
    )
    trainer_arguments = dict(
        learning_rate=0.1,
        momentum=0.0,
        weight_decay=0.0,
        lr_warmup_steps=0,
        unroll_steps=2,
        lstm_horizon=2,
        support_min=0,
        support_max=1,
        max_gradient_norm=100.0,
    )
    eager = MuZeroTrainer(
        representation,
        dynamics,
        prediction,
        **trainer_arguments,
    )
    compiled = MuZeroTrainer(
        *compiled_modules,
        compile_model=True,
        **trainer_arguments,
    )
    batch = ReplayBatch(
        frames=torch.zeros(2, 3, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 2, 1, dtype=torch.long),
        rewards=torch.tensor([[1.0, -0.5], [0.5, 0.25]]),
        policy_targets=torch.tensor(
            [
                [[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]],
                [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]],
            ]
        ),
        value_targets=torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]]),
        action_mask=torch.ones(2, 2, dtype=torch.bool),
        policy_mask=torch.ones(2, 3, dtype=torch.bool),
        value_mask=torch.ones(2, 3, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.tensor([0.5, 1.0]),
    )

    eager_metrics = eager.train_step(batch)
    compiled_metrics = compiled.train_step(batch)

    assert compile_arguments == {
        "dynamic": False,
        "fullgraph": True,
        "mode": "max-autotune",
    }
    for name in (
        "loss",
        "policy_loss",
        "value_loss",
        "reward_loss",
        "gradient_norm",
        "priorities",
    ):
        torch.testing.assert_close(
            getattr(compiled_metrics, name), getattr(eager_metrics, name)
        )
    for eager_parameter, compiled_parameter in zip(
        eager._parameters, compiled._parameters, strict=True
    ):
        torch.testing.assert_close(compiled_parameter, eager_parameter)
        torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad)


def test_muzero_requires_dynamics_gradient_scaling() -> None:
    with pytest.raises(ValueError, match="gradient scaling"):
        MuZeroTrainer(
            _ScalarRepresentation(),
            _IdentityDynamics(scale_state_gradient=False),
            _ScalarPrediction(),
        )


def test_muzero_halves_each_recurrent_state_gradient() -> None:
    representation = _ScalarRepresentation()
    trainer = MuZeroTrainer(
        representation,
        _IdentityDynamics(),
        _ScalarPrediction(),
        learning_rate=0.1,
        lr_warmup_steps=0,
        unroll_steps=2,
        lstm_horizon=2,
        policy_weight=1.0,
        value_weight=0.0,
        reward_weight=0.0,
        support_min=0,
        support_max=1,
    )
    batch = ReplayBatch(
        frames=torch.zeros(2, 3, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 2, 1, dtype=torch.long),
        rewards=torch.zeros(2, 2),
        policy_targets=torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            ]
        ),
        value_targets=torch.zeros(2, 3),
        action_mask=torch.zeros(2, 2, dtype=torch.bool),
        policy_mask=torch.ones(2, 3, dtype=torch.bool),
        value_mask=torch.zeros(2, 3, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
    )

    metrics = trainer.train_step(batch)

    assert metrics.policy_loss == pytest.approx(torch.log(torch.tensor(2.0)))
    assert representation.weight.grad == pytest.approx(-0.1875)


def test_muzero_scales_root_and_recurrent_losses_together() -> None:
    trainer = MuZeroTrainer(
        RepresentationNetwork(4),
        DynamicsNetwork(action_space_size=3),
        PredictionNetwork(action_space_size=3),
        lr_warmup_steps=0,
        unroll_steps=2,
        lstm_horizon=2,
    )
    batch = ReplayBatch(
        frames=torch.randint(0, 256, (2, 6, 1, 96, 96), dtype=torch.uint8),
        actions=torch.zeros(2, 2, 1, dtype=torch.long),
        rewards=torch.zeros(2, 2),
        policy_targets=torch.full((2, 3, 3), 1.0 / 3.0),
        value_targets=torch.zeros(2, 3),
        action_mask=torch.ones(2, 2, dtype=torch.bool),
        policy_mask=torch.ones(2, 3, dtype=torch.bool),
        value_mask=torch.ones(2, 3, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
    )

    metrics = trainer.train_step(batch)

    assert metrics.policy_loss == pytest.approx(
        1.5 * torch.log(torch.tensor(3.0)).item()
    )
    assert metrics.value_loss == pytest.approx(
        1.5 * torch.log(torch.tensor(601.0)).item()
    )
    assert metrics.reward_loss == pytest.approx(
        torch.log(torch.tensor(601.0)).item()
    )


def test_fp16_precision_is_not_supported() -> None:
    with pytest.raises(ValueError, match="fp32 or bf16"):
        MuZeroTrainer(
            RepresentationNetwork(4),
            DynamicsNetwork(action_space_size=3),
            PredictionNetwork(action_space_size=3),
            precision="fp16",  # type: ignore[arg-type]
        )


def test_bf16_precision_requires_cuda() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        MuZeroTrainer(
            RepresentationNetwork(4),
            DynamicsNetwork(action_space_size=3),
            PredictionNetwork(action_space_size=3),
            precision="bf16",
        )


def test_muzero_trainer_uses_efficientzero_v1_optimizer_and_schedule() -> None:
    trainer = MuZeroTrainer(
        RepresentationNetwork(4),
        DynamicsNetwork(action_space_size=3),
        PredictionNetwork(action_space_size=3),
        learning_rate=0.2,
        momentum=0.9,
        weight_decay=1e-4,
        lr_warmup_steps=10,
        lr_decay_rate=0.1,
        lr_decay_steps=100,
    )

    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert trainer.optimizer.defaults["momentum"] == pytest.approx(0.9)
    assert trainer.optimizer.defaults["weight_decay"] == pytest.approx(1e-4)
    assert trainer._adjust_learning_rate() == pytest.approx(0.0)

    trainer._step_count = 5
    assert trainer._adjust_learning_rate() == pytest.approx(0.1)
    trainer._step_count = 10
    assert trainer._adjust_learning_rate() == pytest.approx(0.2)
    trainer._step_count = 110
    assert trainer._adjust_learning_rate() == pytest.approx(0.02)
