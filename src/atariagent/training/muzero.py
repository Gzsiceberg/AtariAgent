"""MuZero-style training on replayed self-play unrolls.

The recurrent reward head predicts EfficientZero value prefixes: rewards are
accumulated between LSTM resets instead of predicting each immediate reward.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn

from atariagent.agent import categorical_to_scalar
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig
from .target import ValueTargetNetwork


Precision = Literal["fp32", "bf16"]


@dataclass(frozen=True, slots=True)
class MuZeroTrainMetrics:
    """Detached metrics from one optimizer update.

    Scalar values stay as tensors so ordinary updates do not synchronize the
    CUDA stream merely to produce logging data. Convert or format them only on
    logging/profiling updates. Replay priorities intentionally remain a tensor
    until the CPU replay update performs its required transfer.
    """

    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    reward_loss: Tensor
    gradient_norm: Tensor
    learning_rate: float
    priorities: Tensor


def _halve_gradient(gradient: Tensor) -> Tensor:
    """Scale recurrent-state gradients as prescribed by MuZero."""
    return gradient * 0.5


class MuZeroTrainer:
    """Train all agent networks from :class:`ReplayBatch` self-play data.

    Root and recurrent losses are summed and scaled by ``1 / unroll_steps``.
    Recurrent latent-state gradients are halved as in MuZero and EfficientZero.
    A periodically hard-copied target network can refresh direct value
    bootstraps and selected policy targets through fresh MCTS. The default Atari
    loss coefficients are policy 1, value 0.25, and value-prefix reward 1.
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
        priority_epsilon: float = 1e-6,
        use_target_network_reanalysis: bool = True,
        policy_reanalysis_ratio: float = 0.0,
        policy_reanalysis_chunk_size: int = 64,
        action_space_size: int | None = None,
        mcts_config: MCTSConfig | None = None,
        reanalysis_seed: int = 0,
        target_update_interval: int = 200,
        precision: Precision = "fp32",
        compile_model: bool = False,
        compile_mode: str = "default",
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
        if priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be positive")
        if not isinstance(use_target_network_reanalysis, bool):
            raise TypeError("use_target_network_reanalysis must be a boolean")
        if isinstance(policy_reanalysis_ratio, bool) or not math.isfinite(
            policy_reanalysis_ratio
        ):
            raise ValueError("policy_reanalysis_ratio must be finite")
        if not 0.0 <= policy_reanalysis_ratio <= 1.0:
            raise ValueError("policy_reanalysis_ratio must be in [0, 1]")
        if (
            isinstance(policy_reanalysis_chunk_size, bool)
            or not isinstance(policy_reanalysis_chunk_size, int)
        ):
            raise TypeError("policy_reanalysis_chunk_size must be an integer")
        if policy_reanalysis_chunk_size <= 0:
            raise ValueError("policy_reanalysis_chunk_size must be positive")
        if policy_reanalysis_ratio > 0.0 and (
            action_space_size is None or action_space_size <= 0
        ):
            raise ValueError(
                "action_space_size must be positive for policy reanalysis"
            )
        if isinstance(reanalysis_seed, bool) or not isinstance(
            reanalysis_seed, int
        ):
            raise TypeError("reanalysis_seed must be an integer")
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if not compile_mode:
            raise ValueError("compile_mode must not be empty")
        for weight, name in (
            (policy_weight, "policy_weight"),
            (value_weight, "value_weight"),
            (reward_weight, "reward_weight"),
        ):
            if weight < 0.0:
                raise ValueError(f"{name} must be non-negative")

        original_modules = (representation, dynamics, prediction)
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
        self.priority_epsilon = priority_epsilon
        self.use_target_network_reanalysis = use_target_network_reanalysis
        self.policy_reanalysis_ratio = policy_reanalysis_ratio
        self.policy_reanalysis_chunk_size = policy_reanalysis_chunk_size
        self.target_update_interval = target_update_interval
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.precision: Precision = precision
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self._device = device
        self._step_count = 0
        self._parameters = parameters
        target_enabled = (
            use_target_network_reanalysis or policy_reanalysis_ratio > 0.0
        )
        self.target_network = (
            ValueTargetNetwork(
                representation,
                prediction,
                dynamics=(dynamics if policy_reanalysis_ratio > 0.0 else None),
                action_space_size=action_space_size,
                mcts_config=mcts_config,
                rng_seed=reanalysis_seed,
                support_min=support_min,
                support_max=support_max,
                precision=precision,
            )
            if target_enabled
            else None
        )
        self.optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        # Compile wrappers share the original parameters. Keeping the original
        # modules on the agent preserves checkpoint state-dict keys.
        if compile_model:
            self.representation = torch.compile(
                representation, dynamic=False, mode=compile_mode
            )
            self.dynamics = torch.compile(
                dynamics, dynamic=False, mode=compile_mode
            )
            self.prediction = torch.compile(
                prediction, dynamic=False, mode=compile_mode
            )

    def train_step(self, batch: ReplayBatch) -> MuZeroTrainMetrics:
        """Run one update from policy, n-step value, and value-prefix targets."""
        self._validate_batch(batch)
        for module in self._original_modules():
            if not module.training:
                module.train()
        learning_rate = self._adjust_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)

        if self.target_network is not None:
            batch = self.target_network.reanalyze_batch(
                batch,
                reanalyze_values=self.use_target_network_reanalysis,
                policy_ratio=self.policy_reanalysis_ratio,
                policy_chunk_size=self.policy_reanalysis_chunk_size,
            )
        observations = batch.normalized_root_observation()
        prefix_targets = batch.value_prefix_targets(
            lstm_horizon=self.lstm_horizon
        )

        autocast_dtype = (
            torch.bfloat16 if self.precision == "bf16" else None
        )
        autocast_context = (
            nullcontext()
            if autocast_dtype is None
            else torch.autocast(
                device_type=self._device.type,
                dtype=autocast_dtype,
            )
        )
        with autocast_context:
            batch_size = batch.batch_size
            recurrent_policy_loss = observations.new_zeros(batch_size)
            recurrent_value_loss = observations.new_zeros(batch_size)
            recurrent_reward_loss = observations.new_zeros(batch_size)

            state = self.representation(observations)
            policy_logits, value_logits = self.prediction(state)
            root_policy_loss, root_value_loss = batch.prediction_losses(
                policy_logits,
                value_logits,
                offset=0,
                support_min=self.support_min,
                support_max=self.support_max,
            )

            # Decode in FP32 even when the model is autocast. This avoids
            # low-precision inverse-transform operations in priorities.
            predicted_root_values = categorical_to_scalar(
                value_logits.detach().float(),
                support_min=self.support_min,
                support_max=self.support_max,
            )
            new_priorities = (
                predicted_root_values - batch.value_targets[:, 0]
            ).abs() + self.priority_epsilon

            hidden = None
            for step in range(self.unroll_steps):
                state, hidden, value_prefix_logits = self.dynamics(
                    state, batch.actions[:, step], hidden
                )
                recurrent_reward_loss += batch.value_prefix_loss(
                    value_prefix_logits,
                    prefix_targets[:, step],
                    step=step,
                    support_min=self.support_min,
                    support_max=self.support_max,
                )

                policy_logits, value_logits = self.prediction(state)
                step_policy_loss, step_value_loss = batch.prediction_losses(
                    policy_logits,
                    value_logits,
                    offset=step + 1,
                    support_min=self.support_min,
                    support_max=self.support_max,
                )
                recurrent_policy_loss += step_policy_loss
                recurrent_value_loss += step_value_loss
                state.register_hook(_halve_gradient)

                if (step + 1) % self.lstm_horizon == 0:
                    hidden = None

            loss_scale = 1.0 / self.unroll_steps
            sample_weights = batch.importance_weights.to(
                root_policy_loss.dtype
            )
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
        if (
            self.target_network is not None
            and self._step_count % self.target_update_interval == 0
        ):
            self.target_network.synchronize(
                self.original_representation,
                self.original_prediction,
                self.original_dynamics,
            )

        return MuZeroTrainMetrics(
            loss=loss.detach(),
            policy_loss=policy_loss.detach(),
            value_loss=value_loss.detach(),
            reward_loss=reward_loss.detach(),
            gradient_norm=gradient_norm.detach(),
            learning_rate=learning_rate,
            priorities=new_priorities.detach(),
        )

    def _original_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.original_representation,
            self.original_dynamics,
            self.original_prediction,
        )

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
    "MuZeroTrainer",
    "MuZeroTrainMetrics",
    "Precision",
]
