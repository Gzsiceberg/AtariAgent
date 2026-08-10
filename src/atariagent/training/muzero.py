"""MuZero-style training on replayed self-play unrolls.

The recurrent reward head predicts EfficientZero value prefixes: rewards are
accumulated between LSTM resets instead of predicting each immediate reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from atariagent.replay import ReplayBatch


@dataclass(frozen=True, slots=True)
class MuZeroTrainMetrics:
    """Scalar metrics from one optimizer update."""

    loss: float
    policy_loss: float
    value_loss: float
    reward_loss: float
    gradient_norm: float
    learning_rate: float


class MuZeroTrainer:
    """Train all agent networks from :class:`ReplayBatch` self-play data.

    Losses are accumulated across the root and recurrent unroll, averaged over
    the batch, and scaled by ``1 / unroll_steps`` as in EfficientZero. The
    three default Atari loss coefficients are policy 1, value 0.25, and
    value-prefix reward 1.
    """

    def __init__(
        self,
        representation: nn.Module,
        dynamics: nn.Module,
        prediction: nn.Module,
        *,
        learning_rate: float = 0.2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        lr_warmup_steps: int = 1_000,
        lr_decay_rate: float = 0.1,
        lr_decay_steps: int = 100_000,
        unroll_steps: int = 5,
        lstm_horizon: int = 5,
        policy_weight: float = 1.0,
        value_weight: float = 0.25,
        reward_weight: float = 1.0,
        max_gradient_norm: float = 5.0,
        support_min: int = -300,
        support_max: int = 300,
    ) -> None:
        if unroll_steps <= 0:
            raise ValueError("unroll_steps must be positive")
        if lstm_horizon <= 0:
            raise ValueError("lstm_horizon must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if momentum < 0.0:
            raise ValueError("momentum must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative")
        if not 0.0 < lr_decay_rate <= 1.0:
            raise ValueError("lr_decay_rate must be in (0, 1]")
        if lr_decay_steps <= 0:
            raise ValueError("lr_decay_steps must be positive")
        if max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        for weight, name in (
            (policy_weight, "policy_weight"),
            (value_weight, "value_weight"),
            (reward_weight, "reward_weight"),
        ):
            if weight < 0.0:
                raise ValueError(f"{name} must be non-negative")

        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
        self.unroll_steps = unroll_steps
        self.lstm_horizon = lstm_horizon
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.reward_weight = reward_weight
        self.max_gradient_norm = max_gradient_norm
        self.support_min = support_min
        self.support_max = support_max
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self._step_count = 0

        modules = (representation, dynamics, prediction)
        parameters = [
            parameter for module in modules for parameter in module.parameters()
        ]
        self._parameters = parameters
        self.optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    def train_step(self, batch: ReplayBatch) -> MuZeroTrainMetrics:
        """Run one update from policy, n-step value, and value-prefix targets."""
        self._validate_batch(batch)
        for module in self._modules():
            module.train()
        learning_rate = self._adjust_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)

        observations = batch.normalized_observations()
        prefix_targets = batch.value_prefix_targets(
            lstm_horizon=self.lstm_horizon
        )
        batch_size = batch.batch_size
        policy_loss = observations.new_zeros(batch_size)
        value_loss = observations.new_zeros(batch_size)
        reward_loss = observations.new_zeros(batch_size)

        state = self.representation(observations[:, 0])
        policy_logits, value_logits = self.prediction(state)
        root_policy_loss, root_value_loss = batch.prediction_losses(
            policy_logits,
            value_logits,
            offset=0,
            support_min=self.support_min,
            support_max=self.support_max,
        )
        policy_loss += root_policy_loss
        value_loss += root_value_loss

        hidden = None
        for step in range(self.unroll_steps):
            state, hidden, value_prefix_logits = self.dynamics(
                state, batch.actions[:, step], hidden
            )

            reward_loss += batch.value_prefix_loss(
                value_prefix_logits,
                prefix_targets[:, step],
                step=step,
                support_min=self.support_min,
                support_max=self.support_max,
            )

            policy_logits, value_logits = self.prediction(state)
            target_offset = step + 1
            step_policy_loss, step_value_loss = batch.prediction_losses(
                policy_logits,
                value_logits,
                offset=target_offset,
                support_min=self.support_min,
                support_max=self.support_max,
            )
            policy_loss += step_policy_loss
            value_loss += step_value_loss

            if (step + 1) % self.lstm_horizon == 0:
                hidden = None

        gradient_scale = 1.0 / self.unroll_steps
        policy_loss = policy_loss.mean() * gradient_scale
        value_loss = value_loss.mean() * gradient_scale
        reward_loss = reward_loss.mean() * gradient_scale
        loss = (
            self.policy_weight * policy_loss
            + self.value_weight * value_loss
            + self.reward_weight * reward_loss
        )
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self._parameters, self.max_gradient_norm
        )
        self.optimizer.step()
        self._step_count += 1

        return MuZeroTrainMetrics(
            loss=float(loss.detach()),
            policy_loss=float(policy_loss.detach()),
            value_loss=float(value_loss.detach()),
            reward_loss=float(reward_loss.detach()),
            gradient_norm=float(gradient_norm.detach()),
            learning_rate=learning_rate,
        )

    def _modules(self) -> tuple[nn.Module, ...]:
        return self.representation, self.dynamics, self.prediction

    def _adjust_learning_rate(self) -> float:
        if self._step_count < self.lr_warmup_steps:
            learning_rate = (
                self.learning_rate * self._step_count / self.lr_warmup_steps
            )
        else:
            decay_count = (
                self._step_count - self.lr_warmup_steps
            ) // self.lr_decay_steps
            learning_rate = self.learning_rate * self.lr_decay_rate**decay_count
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        return learning_rate

    def _validate_batch(self, batch: ReplayBatch) -> None:
        if batch.unroll_steps != self.unroll_steps:
            raise ValueError(
                f"expected {self.unroll_steps} unroll steps, "
                f"got {batch.unroll_steps}"
            )
        if batch.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for batch normalization")
        parameter = next(iter(self._parameters), None)
        if parameter is not None and batch.frames.device != parameter.device:
            raise ValueError("replay batch and training networks must be on one device")
        target_shape = (batch.batch_size, self.unroll_steps + 1)
        if batch.policy_targets.shape[:2] != target_shape:
            raise ValueError("policy_targets has an invalid shape")
        if batch.value_targets.shape != target_shape:
            raise ValueError("value_targets has an invalid shape")
        if batch.value_mask.shape != target_shape:
            raise ValueError("value_mask has an invalid shape")


__all__ = [
    "MuZeroTrainer",
    "MuZeroTrainMetrics",
]
