"""Model-based training on replayed self-play unrolls.

The recurrent reward head predicts EfficientZero value prefixes: rewards are
accumulated between LSTM resets instead of predicting each immediate reward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from atariagent.data import Transforms
from atariagent.models import consist_loss_func
from atariagent.replay_batch import ReplayBatch

Precision = Literal["fp32", "bf16"]
OptimizerName = Literal["sgd", "adam"]


@dataclass(frozen=True, slots=True)
class TrainMetrics:
    """Detached metrics from one optimizer update.

    Scalar values stay as tensors so ordinary updates do not synchronize the
    CUDA stream merely to produce logging data. Convert or format them only on
    logging/profiling updates. Replay priorities intentionally remain a tensor
    until the CPU replay update performs its required transfer.
    """

    # Actual weighted optimization objective.
    loss: Tensor
    # Unweighted means over valid targets, for learning diagnostics.
    policy_loss: Tensor
    value_loss: Tensor  # Mean absolute error of decoded scalar values.
    reward_loss: Tensor  # Same metric for cumulative reward prefixes.
    consistency_loss: Tensor
    behavior_regularization_loss: Tensor
    gradient_norm: Tensor
    search_target_entropy: Tensor
    search_target_max_probability: Tensor
    search_target_effective_actions: Tensor
    importance_weight_mean: Tensor
    importance_weight_ess_fraction: Tensor
    # Across-sample population variance, averaged over latent coordinates.
    representation_feature_variance: Tensor
    dynamics_feature_variance: Tensor
    learning_rate: float
    # Candidate root errors; BatchWorker replaces invalid-root entries with
    # the minimum valid priority (or replay minimum for all-invalid batches).
    priorities: Tensor


class _LearnerUnroll(nn.Module):
    """Complete fixed-length learner forward and loss computation."""

    def __init__(
        self,
        representation: nn.Module,
        dynamics: nn.Module,
        prediction: nn.Module,
        consistency_network: nn.Module,
        augmentation: Transforms | None,
        *,
        observation_dtype: torch.dtype,
        unroll_steps: int,
        policy_weight: float,
        value_weight: float,
        reward_weight: float,
        consistency_weight: float,
        behavior_regularization_weight: float,
        discount: float,
        support_min: int,
        support_max: int,
        priority_epsilon: float,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
        self.consistency_network = consistency_network
        self.augmentation = augmentation
        self.observation_dtype = observation_dtype
        self.unroll_steps = unroll_steps
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.reward_weight = reward_weight
        self.consistency_weight = consistency_weight
        self.support_min = support_min
        self.support_max = support_max
        self.priority_epsilon = priority_epsilon
        self.behavior_regularization_weight = behavior_regularization_weight
        self.discount = discount

    @staticmethod
    def _behavior_loss(
        policy_logits: Tensor,
        actions: Tensor,
        rewards: Tensor,
        values: Tensor,
        next_values: Tensor,
        valid: Tensor,
        discount: float,
    ) -> Tensor:
        """ROSMO's positive-model-advantage filtered behavior cloning."""
        advantage = (rewards + discount * next_values - values).detach()
        selected = valid.bool() & (advantage > 0)
        # Replay actions have shape [batch, 1]; padded actions never contribute.
        actions = rearrange(actions, "batch 1 -> batch").long()
        actions = torch.where(valid.bool(), actions, 0)
        nll = torch.nn.functional.cross_entropy(
            policy_logits.float(), actions, reduction="none"
        )
        return torch.where(selected, nll, 0.0)

    def _prediction_losses(
        self,
        policy_logits: Tensor,
        value_logits: Tensor,
        policy_targets: Tensor,
        value_targets: Tensor,
        value_mask: Tensor,
        offset: int,
    ) -> tuple[Tensor, Tensor]:
        policy_loss = ReplayBatch._policy_cross_entropy(
            policy_logits, policy_targets[:, offset]
        ) * value_mask[:, offset].to(policy_logits.dtype)
        value_loss = ReplayBatch._scalar_loss(
            value_logits,
            value_targets[:, offset],
            support_min=self.support_min,
            support_max=self.support_max,
        ) * value_mask[:, offset].to(value_logits.dtype)
        return policy_loss, value_loss

    @staticmethod
    def _policy_statistics(
        policy_targets: Tensor,
        policy_mask: Tensor,
    ) -> Tensor:
        """Return unweighted sums of search-target diagnostics and valid-root count."""
        targets = policy_targets.detach().float()
        mask = policy_mask.to(dtype=torch.float32)
        target_log_probabilities = torch.where(
            targets > 0.0,
            targets.clamp_min(torch.finfo(torch.float32).tiny).log(),
            torch.zeros_like(targets),
        )
        target_entropy = -(targets * target_log_probabilities).sum(dim=-1)
        return torch.stack(
            (
                (target_entropy * mask).sum(),
                (targets.amax(dim=-1) * mask).sum(),
                (target_entropy.exp() * mask).sum(),
                mask.sum(),
            )
        )

    @staticmethod
    @torch.no_grad()
    def _importance_statistics(weights: Tensor) -> Tensor:
        """Measure loss scaling and weight concentration, without gradients.

        ESS / batch_size measures weight concentration, not the number of
        distinct replay states sampled. Uniform positive weights give one.
        """
        weights = weights.float()
        ess_fraction = weights.sum().square() / (
            weights.square().sum() * weights.numel()
        ).clamp_min(torch.finfo(torch.float32).tiny)
        return torch.stack((weights.mean(), ess_fraction))

    @staticmethod
    @torch.no_grad()
    def _feature_variance(state: Tensor, mask: Tensor) -> Tensor:
        """Mean coordinate-wise variance across valid samples (not space).

        A spatially varying but observation-independent state must report zero.
        Empty and singleton selections have zero population variance. Inputs
        come from the existing augmented training forward; no extra BN updates.
        """
        features = rearrange(state.float(), "batch ... -> batch (...)")
        valid = mask.bool()[:, None]
        count = mask.float().sum().clamp_min(1.0)
        selected = torch.where(valid, features, 0.0)
        mean = selected.sum(dim=0) / count
        centered = torch.where(valid, features - mean, 0.0)
        return (centered.square().sum(dim=0) / count).mean()

    def _decode_values(self, logits: Tensor) -> Tensor:
        """Decode scalar predictions without the eager decoder's cached support."""
        logits = logits.detach().float()
        probabilities = torch.softmax(logits, dim=-1)
        support = torch.arange(
            self.support_min,
            self.support_max + 1,
            device=logits.device,
            dtype=logits.dtype,
        )
        transformed = (probabilities * support).sum(dim=-1)
        epsilon = 0.001
        magnitude = (
            (
                torch.sqrt(
                    1
                    + 4
                    * epsilon
                    * (transformed.abs() + 1 + epsilon)
                )
                - 1
            )
            / (2 * epsilon)
        ).square() - 1
        scalar = torch.nan_to_num(transformed.sign() * magnitude)
        return torch.where(scalar.abs() < epsilon, 0.0, scalar)

    def _prepare_observations(
        self, frames: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Normalize and independently augment root and target observations."""
        batch_size, frame_count, channels, height, width = frames.shape
        stack_size = frame_count - self.unroll_steps
        observations = frames[:, :stack_size].reshape(
            batch_size,
            stack_size * channels,
            height,
            width,
        ).to(dtype=self.observation_dtype).div(255.0)
        if self.augmentation is not None:
            observations = self.augmentation(observations)

        # Packing time into channels gives every target stack one shared
        # spatial shift and intensity draw, independently of the root draw.
        target_sequence = frames[:, 1:].reshape(
            batch_size,
            (frame_count - 1) * channels,
            height,
            width,
        ).to(dtype=self.observation_dtype).div(255.0)
        if self.augmentation is not None:
            target_sequence = self.augmentation(target_sequence)
        target_frames = target_sequence.reshape(
            batch_size,
            frame_count - 1,
            channels,
            height,
            width,
        )
        return observations, target_frames

    def forward(
        self,
        frames: Tensor,
        actions: Tensor,
        rewards: Tensor,
        policy_targets: Tensor,
        value_targets: Tensor,
        action_mask: Tensor,
        importance_weights: Tensor,
    ) -> tuple[Tensor, ...]:
        """Return losses, replay priorities, and policy diagnostics."""
        batch_size = actions.shape[0]
        # V2 has no independent value-validity gate. Real actions supervise
        # their successors (including the zero-target block endpoint).
        value_mask = torch.cat((torch.ones_like(action_mask[:, :1]), action_mask), dim=1)
        policy_mask = (policy_targets.sum(dim=-1) > 0) & value_mask
        stack_size = frames.shape[1] - self.unroll_steps
        observations, target_frames = self._prepare_observations(frames)

        # The LSTM starts fresh for each unroll and never resets inside it.
        prefix_targets = (rewards * action_mask.to(rewards.dtype)).cumsum(dim=1)

        recurrent_policy_loss = observations.new_zeros(batch_size)
        recurrent_value_loss = observations.new_zeros(batch_size)
        recurrent_reward_loss = observations.new_zeros(batch_size)
        recurrent_consistency_loss = observations.new_zeros(batch_size)
        policy_statistics = observations.new_zeros(4, dtype=torch.float32)

        state = self.representation(observations)
        representation_feature_variance = self._feature_variance(
            state, policy_mask[:, 0]
        )
        dynamics_variance_sum = observations.new_zeros((), dtype=torch.float32)
        policy_logits, value_logits = self.prediction(state)
        root_policy_loss, root_value_loss = self._prediction_losses(
            policy_logits,
            value_logits,
            policy_targets,
            value_targets,
            value_mask,
            0,
        )
        policy_statistics += self._policy_statistics(
            policy_targets[:, 0],
            policy_mask[:, 0],
        )
        predicted_root_values = self._decode_values(value_logits)
        value_error = (
            (predicted_root_values - value_targets[:, 0]).abs()
            * value_mask[:, 0]
        ).sum()
        reward_error = observations.new_zeros((), dtype=torch.float32)
        priorities = (
            predicted_root_values - value_targets[:, 0]
        ).abs() + self.priority_epsilon

        behavior_loss = observations.new_zeros(batch_size)
        previous_prefix = torch.zeros_like(predicted_root_values)
        hidden = None
        for step in range(self.unroll_steps):
            if self.behavior_regularization_weight > 0.0:
                behavior_logits = policy_logits
                behavior_values = self._decode_values(value_logits)
            state, hidden, value_prefix_logits = self.dynamics(
                state,
                actions[:, step],
                hidden,
            )
            dynamics_variance_sum += self._feature_variance(
                state, action_mask[:, step]
            ) * action_mask[:, step].sum()
            reward_error += (
                (self._decode_values(value_prefix_logits) - prefix_targets[:, step]).abs()
                * action_mask[:, step]
            ).sum()
            recurrent_reward_loss += ReplayBatch._scalar_loss(
                value_prefix_logits,
                prefix_targets[:, step],
                support_min=self.support_min,
                support_max=self.support_max,
            ) * action_mask[:, step].to(value_prefix_logits.dtype)

            policy_logits, value_logits = self.prediction(state)
            if self.behavior_regularization_weight > 0.0:
                predicted_prefix = self._decode_values(value_prefix_logits)
                # The reward head predicts a prefix over this entire unroll.
                predicted_reward = predicted_prefix - previous_prefix
                behavior_loss += self._behavior_loss(
                    behavior_logits,
                    actions[:, step],
                    predicted_reward,
                    behavior_values,
                    self._decode_values(value_logits),
                    action_mask[:, step],
                    self.discount,
                )
                previous_prefix = predicted_prefix
            step_policy_loss, step_value_loss = self._prediction_losses(
                policy_logits,
                value_logits,
                policy_targets,
                value_targets,
                value_mask,
                step + 1,
            )
            recurrent_policy_loss += step_policy_loss
            recurrent_value_loss += step_value_loss
            value_error += (
                (self._decode_values(value_logits) - value_targets[:, step + 1]).abs()
                * value_mask[:, step + 1]
            ).sum()
            policy_statistics += self._policy_statistics(
                policy_targets[:, step + 1],
                policy_mask[:, step + 1],
            )

            target_observations = target_frames[
                :, step : step + stack_size
            ].reshape(
                batch_size,
                observations.shape[1],
                target_frames.shape[3],
                target_frames.shape[4],
            )
            with torch.no_grad():
                target_state = self.representation(target_observations)
            predicted_projection, target_projection = self.consistency_network(
                state, target_state
            )
            recurrent_consistency_loss += consist_loss_func(
                predicted_projection, target_projection
            ) * action_mask[:, step].to(predicted_projection.dtype)

        loss_scale = 1.0 / self.unroll_steps
        sample_weights = importance_weights.to(root_policy_loss.dtype)
        policy_loss = (
            sample_weights
            * (root_policy_loss + recurrent_policy_loss)
            * loss_scale
        ).mean()
        value_loss = (
            sample_weights
            * (root_value_loss + recurrent_value_loss)
            * loss_scale
        ).mean()
        reward_loss = (
            sample_weights * recurrent_reward_loss * loss_scale
        ).mean()
        consistency_loss = (
            sample_weights * recurrent_consistency_loss * loss_scale
        ).mean()
        loss = (
            self.policy_weight * policy_loss
            + self.value_weight * value_loss
            + self.reward_weight * reward_loss
            + self.consistency_weight * consistency_loss
        )
        if self.behavior_regularization_weight > 0.0:
            loss = loss + self.behavior_regularization_weight * (
                sample_weights * behavior_loss * loss_scale
            ).mean()
        valid_policy_roots = policy_statistics[3].clamp_min(1.0)
        policy_diagnostics = policy_statistics[:3] / valid_policy_roots
        # Logging only: do not let replay weights or padding dilute these means.
        valid_actions = action_mask.sum().clamp_min(1)
        normalized_losses = torch.stack(
            (
                (root_policy_loss + recurrent_policy_loss).detach().float().sum()
                / policy_mask.sum().clamp_min(1),
                value_error / value_mask.sum().clamp_min(1),
                reward_error / valid_actions,
                recurrent_consistency_loss.detach().float().sum() / valid_actions,
                behavior_loss.detach().float().sum() / valid_actions,
            )
        )
        return (
            loss,
            priorities,
            policy_diagnostics[0],
            policy_diagnostics[1],
            policy_diagnostics[2],
            normalized_losses,
            self._importance_statistics(importance_weights),
            representation_feature_variance,
            dynamics_variance_sum / valid_actions,
        )


class Trainer:
    """Train all agent networks from :class:`ReplayBatch` self-play data.

    Root and recurrent losses are summed and scaled by ``1 / unroll_steps``.
    The value-prefix LSTM runs for the same horizon, starting fresh per unroll.
    Recurrent latent-state gradients are halved following EfficientZero.
    Replay batches already contain asynchronously refreshed value and policy
    targets. Recurrent dynamics states are aligned with stop-gradient
    representation states from the corresponding observations. The default
    Atari loss coefficients are policy 1, value 0.25,
    value-prefix reward 1, and consistency 5.
    """

    def __init__(
        self,
        representation: nn.Module,
        dynamics: nn.Module,
        prediction: nn.Module,
        *,
        consistency_network: nn.Module,
        augmentation: Sequence[str] | None = None,
        augmentation_shift_delta: int = 4,
        augmentation_intensity_scale: float = 0.05,
        image_shape: tuple[int, int] = (96, 96),
        optimizer: OptimizerName = "adam",
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        lr_warmup_steps: int = 1_000,
        lr_decay_rate: float = 0.1,
        lr_decay_steps: int = 100_000,
        steps: int = 100_000,
        final_steps: int = 20_000,
        unroll_steps: int = 5,
        policy_weight: float = 1.0,
        value_weight: float = 0.25,
        reward_weight: float = 1.0,
        consistency_weight: float = 5.0,
        behavior_regularization_weight: float = 0.0,
        discount: float = 0.997 ** 4,
        max_gradient_norm: float = 5.0,
        support_min: int = -300,
        support_max: int = 300,
        priority_epsilon: float = 1e-6,
        precision: Precision = "fp32",
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
    ) -> None:
        if (
            not math.isfinite(behavior_regularization_weight)
            or behavior_regularization_weight < 0
        ):
            raise ValueError(
                "behavior_regularization_weight must be finite and non-negative"
            )
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if isinstance(unroll_steps, bool) or not isinstance(unroll_steps, int):
            raise TypeError("unroll_steps must be an integer")
        if unroll_steps <= 0:
            raise ValueError("unroll_steps must be positive")
        if augmentation_shift_delta < 0:
            raise ValueError("augmentation_shift_delta must be non-negative")
        if augmentation_intensity_scale < 0.0:
            raise ValueError("augmentation_intensity_scale must be non-negative")
        if optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be sgd or adam")
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
        if steps <= 0:
            raise ValueError("steps must be positive")
        if final_steps < 0:
            raise ValueError("final_steps must be non-negative")
        if max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        if priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be positive")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if not compile_mode:
            raise ValueError("compile_mode must not be empty")
        if getattr(dynamics, "scale_state_gradient", None) is not True:
            raise ValueError(
                "dynamics must enable recurrent-state gradient scaling"
            )
        for weight, name in (
            (policy_weight, "policy_weight"),
            (value_weight, "value_weight"),
            (reward_weight, "reward_weight"),
            (consistency_weight, "consistency_weight"),
        ):
            if weight < 0.0:
                raise ValueError(f"{name} must be non-negative")

        original_modules = (
            representation,
            dynamics,
            prediction,
            consistency_network,
        )
        parameters = [
            parameter
            for module in original_modules
            for parameter in module.parameters()
        ]
        parameter = next(iter(parameters), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        if precision == "bf16" and device.type != "cuda":
            raise ValueError("bf16 learner precision requires a CUDA device")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bf16 learner precision is unsupported by the selected CUDA device"
            )

        self.original_representation = representation
        self.original_dynamics = dynamics
        self.original_prediction = prediction
        self.original_consistency_network = consistency_network
        self.transforms = (
            None
            if augmentation is None
            else Transforms(
                augmentation,
                shift_delta=augmentation_shift_delta,
                image_shape=image_shape,
                intensity_scale=augmentation_intensity_scale,
            )
        )
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
        self.consistency_network = consistency_network
        self._unroll: nn.Module
        self.unroll_steps = unroll_steps
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.reward_weight = reward_weight
        self.consistency_weight = consistency_weight
        self.max_gradient_norm = max_gradient_norm
        self.support_min = support_min
        self.support_max = support_max
        self.priority_epsilon = priority_epsilon
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.steps = steps
        self.final_steps = final_steps
        self.precision: Precision = precision
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self._device = device
        self._step_count = 0
        self._parameters = parameters
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        # Eager and compiled training share exactly one unroll implementation.
        # The original modules retain checkpoint state-dict keys because the
        # unroll and its compile wrapper reference the same parameters.
        unroll = _LearnerUnroll(
            representation,
            dynamics,
            prediction,
            consistency_network,
            self.transforms,
            # Normalize and augment in FP32, independently of network autocast.
            # In particular, intensity noise must also be sampled in FP32.
            observation_dtype=torch.float32,
            unroll_steps=unroll_steps,
            policy_weight=policy_weight,
            value_weight=value_weight,
            reward_weight=reward_weight,
            consistency_weight=consistency_weight,
            behavior_regularization_weight=behavior_regularization_weight,
            discount=discount,
            support_min=support_min,
            support_max=support_max,
            priority_epsilon=priority_epsilon,
        )
        self._uncompiled_unroll = unroll
        if compile_model:
            torch._dynamo.config.allow_rnn = True
            self._unroll = torch.compile(
                unroll,
                dynamic=False,
                fullgraph=True,
                mode=compile_mode,
            )
        else:
            self._unroll = unroll

    @property
    def step_count(self) -> int:
        """Return the number of optimizer updates already applied."""
        return self._step_count

    def training_state_dict(self) -> dict[str, object]:
        """Return optimizer and schedule state needed to resume training."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
        }

    def load_training_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore optimizer and learning-rate schedule state."""
        if not isinstance(state, Mapping):
            raise TypeError("trainer state must be a mapping")
        step_count = state.get("step_count")
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise TypeError("trainer step_count must be an integer")
        if step_count < 0:
            raise ValueError("trainer step_count must be non-negative")
        optimizer_state = state.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise TypeError("trainer optimizer state must be a mapping")
        self.optimizer.load_state_dict(dict(optimizer_state))
        self._step_count = step_count

    def train_step(self, batch: ReplayBatch) -> TrainMetrics:
        """Run one update from policy, n-step value, and value-prefix targets."""
        self._validate_batch(batch)
        for module in self._original_modules():
            if not module.training:
                module.train()
        learning_rate = self._adjust_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)

        autocast_dtype = torch.bfloat16 if self.precision == "bf16" else None
        autocast_context = (
            nullcontext()
            if autocast_dtype is None
            else torch.autocast(
                device_type=self._device.type,
                dtype=autocast_dtype,
            )
        )
        with autocast_context:
            outputs = self._unroll(
                batch.frames,
                batch.actions,
                batch.rewards,
                batch.policy_targets,
                batch.value_targets,
                batch.action_mask,
                batch.importance_weights,
            )
            (
                loss,
                new_priorities,
                search_target_entropy,
                search_target_max_probability,
                search_target_effective_actions,
                normalized_losses,
                importance_statistics,
                representation_feature_variance,
                dynamics_feature_variance,
            ) = outputs

        loss.backward()

        gradient_norm = nn.utils.clip_grad_norm_(
            self._parameters, self.max_gradient_norm
        )

        self.optimizer.step()
        self._step_count += 1

        return TrainMetrics(
            loss=loss.detach(),
            policy_loss=normalized_losses[0].detach(),
            value_loss=normalized_losses[1].detach(),
            reward_loss=normalized_losses[2].detach(),
            consistency_loss=normalized_losses[3].detach(),
            behavior_regularization_loss=normalized_losses[4].detach(),
            gradient_norm=gradient_norm.detach(),
            search_target_entropy=search_target_entropy.detach(),
            search_target_max_probability=(
                search_target_max_probability.detach()
            ),
            search_target_effective_actions=(
                search_target_effective_actions.detach()
            ),
            importance_weight_mean=importance_statistics[0],
            importance_weight_ess_fraction=importance_statistics[1],
            representation_feature_variance=representation_feature_variance,
            dynamics_feature_variance=dynamics_feature_variance,
            learning_rate=learning_rate,
            priorities=new_priorities.detach(),
        )

    @torch.no_grad()
    def _prepare_observations(
        self, frames: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Prepare observations through the same path as the learner graph."""
        return self._uncompiled_unroll._prepare_observations(frames)

    def _original_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.original_representation,
            self.original_dynamics,
            self.original_prediction,
            self.original_consistency_network,
        )

    def _adjust_learning_rate(self) -> float:
        if self.optimizer_name == "sgd":
            if self._step_count < self.lr_warmup_steps:
                learning_rate = (
                    self.learning_rate * self._step_count / self.lr_warmup_steps
                )
            else:
                decay_count = (
                    self._step_count - self.lr_warmup_steps
                ) // self.lr_decay_steps
                learning_rate = (
                    self.learning_rate * self.lr_decay_rate**decay_count
                )
        else:
            decay_progress = (
                float(self._step_count >= self.steps)
                if self.final_steps == 0
                else min(
                    max(self._step_count - self.steps, 0) / self.final_steps,
                    1.0,
                )
            )
            learning_rate = self.learning_rate * (
                1.0 - decay_progress * (1.0 - self.lr_decay_rate)
            )
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
        bootstrap_metadata = (
            batch.value_bootstrap_frames,
            batch.value_bootstrap_values,
            batch.value_bootstrap_discounts,
            batch.value_bootstrap_mask,
        )
        has_bootstrap_metadata = tuple(
            value is not None for value in bootstrap_metadata
        )
        if any(has_bootstrap_metadata) and not all(has_bootstrap_metadata):
            raise ValueError("value-bootstrap metadata is incomplete")
        if all(has_bootstrap_metadata):
            assert batch.value_bootstrap_frames is not None
            assert batch.value_bootstrap_values is not None
            assert batch.value_bootstrap_discounts is not None
            assert batch.value_bootstrap_mask is not None
            if batch.value_bootstrap_frames.shape != batch.frames.shape:
                raise ValueError("value_bootstrap_frames has an invalid shape")
            for value in (
                batch.value_bootstrap_values,
                batch.value_bootstrap_discounts,
                batch.value_bootstrap_mask,
            ):
                if value.shape != target_shape:
                    raise ValueError("value-bootstrap metadata has an invalid shape")
        if batch.indices.shape != (batch.batch_size,):
            raise ValueError("indices has an invalid shape")
        if batch.importance_weights.shape != (batch.batch_size,):
            raise ValueError("importance_weights has an invalid shape")


__all__ = [
    "Precision",
    "TrainMetrics",
    "Trainer",
]
