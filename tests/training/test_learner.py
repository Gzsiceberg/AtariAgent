from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as functional

from atariagent.models import (
    ConsistencyNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.training import Trainer


class _ZeroConsistency(torch.nn.Module):
    """Keep consistency mandatory without affecting unrelated loss tests."""

    def forward(
        self,
        predicted_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = predicted_state.flatten(1)[:, :1] * 0.0
        target = target_state.flatten(1)[:, :1].detach() * 0.0
        return prediction, target


def test_scalar_categorical_loss_interpolates_transformed_target() -> None:
    logits = torch.randn(2, 601)
    targets = torch.tensor([0.0, 1.0])
    loss = ReplayBatch._scalar_loss(
        logits,
        targets,
        support_min=-300,
        support_max=300,
    )
    transformed = (
        targets.sign() * (torch.sqrt(targets.abs() + 1.0) - 1.0) + 0.001 * targets + 300
    )
    lower = transformed.floor().long()
    upper = transformed.ceil().long()
    upper_weight = transformed - lower
    expected = (1.0 - upper_weight) * functional.cross_entropy(
        logits, lower, reduction="none"
    ) + upper_weight * functional.cross_entropy(logits, upper, reduction="none")

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
        targets.sign() * (torch.sqrt(targets.abs() + 1.0) - 1.0) + 0.001 * targets
    ).clamp(-300, 300) + 300
    lower = transformed.floor().long()
    upper = transformed.ceil().long()
    upper_weight = transformed - lower
    expected = (1.0 - upper_weight) * functional.cross_entropy(
        reference_logits, lower, reduction="none"
    ) + upper_weight * functional.cross_entropy(
        reference_logits, upper, reduction="none"
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


def test_agent_train_step_updates_all_supervised_output_heads() -> None:
    representation = RepresentationNetwork(4)
    dynamics = DynamicsNetwork(action_space_size=3)
    prediction = PredictionNetwork(action_space_size=3)
    consistency = ConsistencyNetwork(
        projection_dim=32,
        projection_hidden_dim=64,
        prediction_hidden_dim=16,
    )
    trainer = Trainer(
        representation,
        dynamics,
        prediction,
        consistency_network=consistency,
        unroll_steps=1,
    )
    policy_output = prediction.policy.projection[-1]
    value_output = prediction.value.projection[-1]
    reward_output = dynamics.reward_prediction.projection[-1]
    consistency_output = consistency.predictor.network[-1]
    assert not hasattr(trainer, "target_network")
    initial_policy = policy_output.weight.detach().clone()
    initial_value = value_output.weight.detach().clone()
    initial_reward = reward_output.weight.detach().clone()
    initial_consistency = consistency_output.weight.detach().clone()

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
        2.0 * metrics.policy_loss
        + 0.25 * 2.0 * torch.log(torch.tensor(601.0))
        + torch.log(torch.tensor(601.0))
        + 5.0 * metrics.consistency_loss
    )
    assert metrics.policy_loss > 0.0
    assert metrics.value_loss > 0.0
    assert metrics.reward_loss > 0.0
    assert torch.isfinite(metrics.consistency_loss)
    assert metrics.consistency_loss.abs() > 0.0
    assert metrics.learning_rate == pytest.approx(0.001)
    assert metrics.priorities == pytest.approx((1.000001, 1.000001))
    assert metrics.importance_weight_mean.item() == pytest.approx(1.0)
    assert metrics.importance_weight_ess_fraction.item() == pytest.approx(1.0)
    for name in (
        "importance_weight_mean", "importance_weight_ess_fraction",
        "representation_feature_variance", "dynamics_feature_variance",
    ):
        diagnostic = getattr(metrics, name)
        assert not diagnostic.requires_grad
        assert diagnostic.ndim == 0
        assert torch.isfinite(diagnostic)
        assert diagnostic >= 0
    assert not torch.equal(policy_output.weight, initial_policy)
    assert not torch.equal(value_output.weight, initial_value)
    assert not torch.equal(reward_output.weight, initial_reward)
    assert not torch.equal(consistency_output.weight, initial_consistency)


class _RecordingTransforms(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.inputs.append(images.clone())
        return images + len(self.inputs)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_trainer_augments_root_and_packed_target_sequence_separately(precision) -> None:
    if precision == "bf16" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("BF16 learner requires a supported CUDA device")
    device = torch.device("cuda" if precision == "bf16" else "cpu")
    trainer = Trainer(
        RepresentationNetwork(2).to(device),
        DynamicsNetwork(action_space_size=3).to(device),
        PredictionNetwork(action_space_size=3).to(device),
        consistency_network=ConsistencyNetwork(
            projection_dim=32,
            projection_hidden_dim=64,
            prediction_hidden_dim=16,
        ),
        augmentation=["none"],
        image_shape=(2, 2),
        unroll_steps=2,
        precision=precision,
    )
    recorder = _RecordingTransforms()
    trainer._uncompiled_unroll.augmentation = recorder
    frame_values = torch.arange(4, dtype=torch.uint8).reshape(
        1, 4, 1, 1, 1
    )
    frames = frame_values.expand(2, 4, 1, 2, 2).clone().to(device)

    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
    ):
        observations, targets = trainer._prepare_observations(frames)
        # Network operations still use BF16 under the enclosing autocast.
        probe = torch.nn.functional.linear(observations.flatten(1), torch.ones(1, 8, device=device))
        assert probe.dtype == (torch.bfloat16 if precision == "bf16" else torch.float32)

    assert all(value.dtype == torch.float32 for value in recorder.inputs)
    assert observations.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert targets is not None
    assert [value.shape for value in recorder.inputs] == [
        (2, 2, 2, 2),
        (2, 3, 2, 2),
    ]
    expected_root = (
        frames[:, :2].reshape(2, 2, 2, 2).float() / 255.0 + 1.0
    )
    expected_targets = frames[:, 1:].float() / 255.0 + 2.0
    torch.testing.assert_close(observations, expected_root)
    torch.testing.assert_close(targets, expected_targets)


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
    consistency = _ZeroConsistency()
    compiled_modules = tuple(
        deepcopy(module)
        for module in (representation, dynamics, prediction, consistency)
    )
    trainer_arguments = dict(
        learning_rate=0.1,
        momentum=0.0,
        weight_decay=0.0,
        unroll_steps=2,
        support_min=0,
        support_max=1,
        max_gradient_norm=100.0,
    )
    eager = Trainer(
        representation,
        dynamics,
        prediction,
        consistency_network=consistency,
        **trainer_arguments,
    )
    compiled = Trainer(
        *compiled_modules[:3],
        consistency_network=compiled_modules[3],
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
        "consistency_loss",
        "gradient_norm",
        "search_target_entropy",
        "search_target_max_probability",
        "search_target_effective_actions",
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


class _AlignedConsistency(torch.nn.Module):
    def forward(self, predicted_state, target_state):
        return predicted_state * 0.0 + 1.0, target_state.detach() * 0.0 + 1.0


@pytest.mark.parametrize("importance_weight", [1.0, 0.01])
@pytest.mark.parametrize("valid_steps", [0, 1, 2])
def test_logged_losses_ignore_importance_weights_and_padding(
    importance_weight: float, valid_steps: int
) -> None:
    trainer = Trainer(
        _ScalarRepresentation(),
        _IdentityDynamics(),
        _ScalarPrediction(),
        consistency_network=_AlignedConsistency(),
        unroll_steps=2,
        support_min=0,
        support_max=1,
    )
    action_mask = torch.arange(2).expand(2, -1) < valid_steps
    policy_mask = torch.arange(3).expand(2, -1) < valid_steps
    batch = ReplayBatch(
        frames=torch.zeros(2, 3, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 2, 1, dtype=torch.long),
        rewards=torch.zeros(2, 2),
        policy_targets=torch.full((2, 3, 2), 0.5) * policy_mask[..., None],
        value_targets=torch.zeros(2, 3),
        action_mask=action_mask,
        indices=torch.arange(2),
        importance_weights=torch.full((2,), importance_weight),
    )
    metrics = trainer.train_step(batch)
    log_two = torch.log(torch.tensor(2.0)).item()
    # Uniform logits on support [0, 1] decode from transformed scalar 0.5.
    decoded = (((1 + 4 * 0.001 * (0.5 + 1 + 0.001)) ** 0.5 - 1)
               / (2 * 0.001)) ** 2 - 1
    for name, count, expected in (
        ("policy", valid_steps, log_two),
        ("value", valid_steps, abs(decoded)),
        ("reward", valid_steps, abs(decoded)),
        ("consistency", valid_steps, -1.0),
    ):
        normalized = getattr(metrics, f"{name}_loss")
        assert not normalized.requires_grad
        assert normalized.item() == pytest.approx(
            expected if count else 0.0, rel=1e-3
        )
    # The optimization objective still uses weights and fixed unroll scaling.
    expected_loss = importance_weight / 2 * (
        valid_steps * log_two
        + 0.25 * valid_steps * log_two
        + valid_steps * log_two
        - 5.0 * valid_steps
    )
    assert metrics.loss.item() == pytest.approx(expected_loss)


def test_padding_mask_blocks_all_recurrent_target_gradients_after_last_action() -> None:
    def trainer():
        return Trainer(
            _ScalarRepresentation(), _RewardIdentityDynamics(), _ScalarPrediction(),
            consistency_network=_AlignedConsistency(), unroll_steps=2,
            support_min=0, support_max=1,
        )

    batch = ReplayBatch(
        frames=torch.zeros(2, 3, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 2, 1, dtype=torch.long),
        rewards=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        policy_targets=torch.tensor([[[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]]] * 2),
        value_targets=torch.tensor([[1.0, 0.0, 0.0]] * 2),
        action_mask=torch.tensor([[True, False]] * 2),
        indices=torch.arange(2), importance_weights=torch.ones(2),
    )
    # State 1 is an unsupervised prediction endpoint. State 2/action 1 are
    # padding: even nonzero policy targets there must not affect any gradient.
    policies = batch.policy_targets.clone()
    policies[:, 2] = torch.tensor([0.0, 1.0])
    poisoned = replace(
        batch, policy_targets=policies,
        value_targets=torch.tensor([[1.0, 0.0, 123.0]] * 2),
        rewards=torch.tensor([[1.0, 123.0]] * 2),
    )
    clean, dirty = trainer(), trainer()
    clean_metrics, dirty_metrics = clean.train_step(batch), dirty.train_step(poisoned)
    torch.testing.assert_close(clean_metrics.loss, dirty_metrics.loss)
    for first, second in zip(clean._parameters, dirty._parameters, strict=True):
        torch.testing.assert_close(first.grad, second.grad)
        torch.testing.assert_close(first, second)


@pytest.mark.parametrize("recorded_steps", [1, 3, 5])
def test_lookahead_has_reward_and_consistency_gradients_but_no_prediction_loss(
    recorded_steps: int,
) -> None:
    class RecordingPrediction(_ScalarPrediction):
        def __init__(self):
            super().__init__()
            self.outputs = []

        def forward(self, state):
            policy, value = super().forward(state)
            policy.retain_grad()
            value.retain_grad()
            self.outputs.append((policy, value))
            return policy, value

    class RecordingDynamics(_RewardIdentityDynamics):
        def __init__(self):
            super().__init__()
            self.outputs = []
            self.actions = []

        def forward(self, state, action, hidden):
            state, hidden, reward = super().forward(state, action, hidden)
            reward.retain_grad()
            self.outputs.append(reward)
            self.actions.append(action.detach().clone())
            return state, hidden, reward

    class RecordingConsistency(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.outputs = []

        def forward(self, state, target):
            projection = torch.cat((state, torch.ones_like(state)), dim=1)
            target_projection = torch.cat((torch.ones_like(target), torch.zeros_like(target)), dim=1)
            projection.retain_grad()
            self.outputs.append(projection)
            return projection, target_projection.detach()

    prediction, dynamics, consistency = (
        RecordingPrediction(), RecordingDynamics(), RecordingConsistency()
    )
    trainer = Trainer(
        _ScalarRepresentation(), dynamics, prediction,
        consistency_network=consistency, unroll_steps=5,
        support_min=0, support_max=1,
    )
    # The root is the last state in its block (100). Successors 101..105
    # have no policy/value targets, even when real lookahead is available.
    policies = torch.zeros(2, 6, 2)
    policies[:, 0] = torch.tensor([1.0, 0.0])
    batch = ReplayBatch(
        frames=torch.zeros(2, 6, 1, 1, 1, dtype=torch.uint8),
        actions=torch.tensor([[[0], [1], [0], [1], [1]]] * 2),
        rewards=torch.full((2, 5), 0.25),
        policy_targets=policies,
        # Poison the unsupervised values to catch accidental endpoint loss.
        value_targets=torch.tensor([[0.0, 100.0, 100.0, 100.0, 100.0, 100.0]] * 2),
        action_mask=(torch.arange(5)[None, :] < recorded_steps).expand(2, -1),
        indices=torch.arange(2), importance_weights=torch.ones(2),
    )
    trainer.train_step(batch)
    for offset, (policy, value) in enumerate(prediction.outputs):
        assert bool(policy.grad.abs().sum() > 0) == (offset == 0)
        assert bool(value.grad.abs().sum() > 0) == (offset == 0)
    for step, (reward, projection) in enumerate(zip(
        dynamics.outputs, consistency.outputs, strict=True
    )):
        assert bool(reward.grad.abs().sum() > 0) == (step < recorded_steps)
        assert bool(projection.grad.abs().sum() > 0) == (step < recorded_steps)
        torch.testing.assert_close(dynamics.actions[step], batch.actions[:, step])


def test_agent_requires_dynamics_gradient_scaling() -> None:
    with pytest.raises(ValueError, match="gradient scaling"):
        Trainer(
            _ScalarRepresentation(),
            _IdentityDynamics(scale_state_gradient=False),
            _ScalarPrediction(),
            consistency_network=_ZeroConsistency(),
        )


def test_agent_halves_each_recurrent_state_gradient() -> None:
    representation = _ScalarRepresentation()
    trainer = Trainer(
        representation,
        _IdentityDynamics(),
        _ScalarPrediction(),
        consistency_network=_ZeroConsistency(),
        learning_rate=0.1,
        unroll_steps=2,
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
        action_mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
    )

    metrics = trainer.train_step(batch)

    assert metrics.policy_loss == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )
    assert representation.weight.grad == pytest.approx(-0.1875)


def test_agent_logs_mean_absolute_error() -> None:
    trainer = Trainer(
        RepresentationNetwork(4),
        DynamicsNetwork(action_space_size=3),
        PredictionNetwork(action_space_size=3),
        consistency_network=_ZeroConsistency(),
        unroll_steps=2,
    )
    batch = ReplayBatch(
        frames=torch.randint(0, 256, (2, 6, 1, 96, 96), dtype=torch.uint8),
        actions=torch.zeros(2, 2, 1, dtype=torch.long),
        rewards=torch.tensor([[1.0, 2.0], [-1.0, -2.0]]),
        policy_targets=torch.full((2, 3, 3), 1.0 / 3.0),
        value_targets=torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
        action_mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(2),
        importance_weights=torch.ones(2),
    )

    metrics = trainer.train_step(batch)

    assert metrics.policy_loss == pytest.approx(
        torch.log(torch.tensor(3.0)).item()
    )
    # Uniform symmetric-support logits decode to zero before the update.
    # Positive and negative targets contribute equally to MAE.
    assert metrics.value_loss == pytest.approx(2.0)
    assert metrics.reward_loss == pytest.approx(2.0)
    assert metrics.consistency_loss == 0.0
    assert metrics.search_target_entropy == pytest.approx(
        torch.log(torch.tensor(3.0)).item()
    )
    assert metrics.search_target_max_probability == pytest.approx(1.0 / 3.0)
    assert metrics.search_target_effective_actions == pytest.approx(3.0)


def test_fp16_precision_is_not_supported() -> None:
    with pytest.raises(ValueError, match="fp32 or bf16"):
        Trainer(
            RepresentationNetwork(4),
            DynamicsNetwork(action_space_size=3),
            PredictionNetwork(action_space_size=3),
            consistency_network=_ZeroConsistency(),
            precision="fp16",  # type: ignore[arg-type]
        )


def test_bf16_precision_requires_cuda() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        Trainer(
            RepresentationNetwork(4),
            DynamicsNetwork(action_space_size=3),
            PredictionNetwork(action_space_size=3),
            consistency_network=_ZeroConsistency(),
            precision="bf16",
        )


def test_agent_trainer_defaults_to_adam() -> None:
    trainer = Trainer(
        RepresentationNetwork(4),
        DynamicsNetwork(action_space_size=3),
        PredictionNetwork(action_space_size=3),
        consistency_network=_ZeroConsistency(),
    )

    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer.defaults["lr"] == pytest.approx(0.001)
    assert trainer.optimizer.defaults["weight_decay"] == pytest.approx(1e-4)
    assert trainer._adjust_learning_rate() == pytest.approx(0.001)

    trainer._step_count = 100_000
    assert trainer._adjust_learning_rate() == pytest.approx(0.001)
    trainer._step_count = 110_000
    assert trainer._adjust_learning_rate() == pytest.approx(0.00055)
    trainer._step_count = 120_000
    assert trainer._adjust_learning_rate() == pytest.approx(0.0001)


def test_agent_trainer_preserves_sgd_and_step_learning_rate_schedule() -> None:
    trainer = Trainer(
        RepresentationNetwork(4),
        DynamicsNetwork(action_space_size=3),
        PredictionNetwork(action_space_size=3),
        consistency_network=_ZeroConsistency(),
        optimizer="sgd",
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
