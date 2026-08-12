"""Tensor batch representation sampled from trajectory replay."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from jaxtyping import Bool, Float, Int, UInt8
import torch
import torch.nn.functional as functional
from torch import Tensor

from .typecheck import runtime_typed


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A padded EfficientZero-style unroll batch.

    ``frames`` contains the initial stack context followed by one new frame
    per unroll action. Call :meth:`normalized_observations` to reconstruct the
    overlapping state stacks for training. ``action_mask`` identifies real
    action/reward steps. ``target_mask`` identifies states with stored search
    targets. ``value_mask`` identifies states whose fixed-horizon return can
    be computed; a true terminal state has a valid zero-value target without
    an MCTS policy. Value-bootstrap fields carry compact real observations and
    stored bootstrap terms so the learner can substitute fresh target-network
    values without changing stored replay data.
    """

    frames: UInt8[Tensor, "batch frames channels height width"]
    actions: Int[Tensor, "batch unroll 1"]
    rewards: Float[Tensor, "batch unroll"]
    policy_targets: Float[Tensor, "batch states actions"]
    value_targets: Float[Tensor, "batch states"]
    action_mask: Bool[Tensor, "batch unroll"]
    target_mask: Bool[Tensor, "batch states"]
    value_mask: Bool[Tensor, "batch states"]
    indices: Int[Tensor, "batch"]
    importance_weights: Float[Tensor, "batch"]
    value_bootstrap_frames: (
        UInt8[Tensor, "batch bootstrap_frames channels height width"] | None
    ) = None
    value_bootstrap_values: Float[Tensor, "batch states"] | None = None
    value_bootstrap_discounts: Float[Tensor, "batch states"] | None = None
    value_bootstrap_mask: Bool[Tensor, "batch states"] | None = None

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def unroll_steps(self) -> int:
        return self.actions.shape[1]

    @property
    def stack_size(self) -> int:
        return self.frames.shape[1] - self.unroll_steps

    @property
    def policy_mask(self) -> Bool[Tensor, "batch states"]:
        """Identify states with a real stored self-play policy target."""
        return self.target_mask & self.policy_targets.sum(dim=-1).gt(0.0)

    @runtime_typed
    def normalized_root_observation(
        self,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> Float[Tensor, "batch stacked_channels height width"]:
        """Reconstruct and normalize only the initial frame stack."""
        batch_size, _, channels, height, width = self.frames.shape
        root = self.frames[:, : self.stack_size].reshape(
            batch_size,
            self.stack_size * channels,
            height,
            width,
        )
        return root.to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).div_(255.0)

    @runtime_typed
    def stacked_observations(
        self,
    ) -> UInt8[Tensor, "batch states stacked_channels height width"]:
        """Reconstruct all overlapping channel-first state stacks."""
        batch_size, _, channels, height, width = self.frames.shape
        return torch.stack(
            tuple(
                self.frames[:, offset : offset + self.stack_size].reshape(
                    batch_size,
                    self.stack_size * channels,
                    height,
                    width,
                )
                for offset in range(self.unroll_steps + 1)
            ),
            dim=1,
        )

    @runtime_typed
    def normalized_observations(
        self, device: torch.device | str | None = None
    ) -> Float[Tensor, "batch states stacked_channels height width"]:
        """Return reconstructed state stacks as floats in ``[0, 1]``."""
        return self.stacked_observations().to(
            device=device, dtype=torch.float32
        ).div_(255.0)

    def normalized_value_bootstrap_observation(
        self,
        offset: int,
    ) -> Float[Tensor, "batch stacked_channels height width"]:
        """Return actual observations used for target-network bootstrapping."""
        if self.value_bootstrap_frames is None:
            raise ValueError("batch has no value-bootstrap observations")
        if not 0 <= offset <= self.unroll_steps:
            raise ValueError("bootstrap offset is outside the unroll")
        batch_size, _, channels, height, width = (
            self.value_bootstrap_frames.shape
        )
        observation = self.value_bootstrap_frames[
            :, offset : offset + self.stack_size
        ].reshape(
            batch_size,
            self.stack_size * channels,
            height,
            width,
        )
        return observation.to(dtype=torch.float32).div_(255.0)

    def with_reanalyzed_value_targets(
        self,
        fresh_bootstrap_values: Float[Tensor, "batch states"],
    ) -> ReplayBatch:
        """Replace stored MCTS bootstraps with fresh target-network values."""
        metadata = (
            self.value_bootstrap_values,
            self.value_bootstrap_discounts,
            self.value_bootstrap_mask,
        )
        if any(value is None for value in metadata):
            raise ValueError("batch has no value-bootstrap metadata")
        if fresh_bootstrap_values.shape != self.value_targets.shape:
            raise ValueError("fresh bootstrap values have an invalid shape")
        assert self.value_bootstrap_values is not None
        assert self.value_bootstrap_discounts is not None
        assert self.value_bootstrap_mask is not None
        bootstrap_delta = (
            fresh_bootstrap_values - self.value_bootstrap_values
        ) * self.value_bootstrap_discounts
        targets = torch.where(
            self.value_bootstrap_mask,
            self.value_targets + bootstrap_delta,
            self.value_targets,
        )
        return replace(self, value_targets=targets)

    def with_reanalysis_targets(
        self,
        *,
        value_targets: Float[Tensor, "batch states"],
        policy_targets: Float[Tensor, "batch states actions"],
    ) -> ReplayBatch:
        """Merge target-only asynchronous reanalysis output into this batch."""
        if value_targets.shape != self.value_targets.shape:
            raise ValueError("reanalyzed value targets have an invalid shape")
        if policy_targets.shape != self.policy_targets.shape:
            raise ValueError("reanalyzed policy targets have an invalid shape")
        if value_targets.device != self.value_targets.device:
            raise ValueError("reanalyzed value targets are on the wrong device")
        if policy_targets.device != self.policy_targets.device:
            raise ValueError("reanalyzed policy targets are on the wrong device")
        return replace(
            self,
            value_targets=value_targets,
            policy_targets=policy_targets,
        )

    def with_reanalyzed_policy_targets(
        self,
        fresh_policy_targets: Float[Tensor, "batch states actions"],
    ) -> ReplayBatch:
        """Use fresh policies for states with valid policy targets."""
        if fresh_policy_targets.shape != self.policy_targets.shape:
            raise ValueError("fresh policy targets have an invalid shape")
        policy_targets = torch.where(
            self.policy_mask[:, :, None],
            fresh_policy_targets,
            self.policy_targets,
        )
        return replace(self, policy_targets=policy_targets)

    @runtime_typed
    def prediction_losses(
        self,
        policy_logits: Float[Tensor, "batch actions"],
        value_logits: Float[Tensor, "batch support"],
        *,
        offset: int,
        support_min: int = -300,
        support_max: int = 300,
    ) -> tuple[
        Float[Tensor, "batch"],
        Float[Tensor, "batch"],
    ]:
        """Return masked policy and n-step value losses for one state."""
        policy_target = self.policy_targets[:, offset]
        has_policy = policy_target.sum(dim=-1) > 0.0
        policy_mask = self.target_mask[:, offset] & has_policy
        policy_loss = self._policy_cross_entropy(
            policy_logits, policy_target
        ) * policy_mask.to(policy_logits.dtype)

        value_loss = self._scalar_loss(
            value_logits,
            self.value_targets[:, offset],
            support_min=support_min,
            support_max=support_max,
        ) * self.value_mask[:, offset].to(value_logits.dtype)
        return policy_loss, value_loss

    @runtime_typed
    def value_prefix_loss(
        self,
        logits: Float[Tensor, "batch support"],
        target: Float[Tensor, "batch"],
        *,
        step: int,
        support_min: int = -300,
        support_max: int = 300,
    ) -> Float[Tensor, "batch"]:
        """Return masked categorical value-prefix loss for one action step."""
        return self._scalar_loss(
            logits,
            target,
            support_min=support_min,
            support_max=support_max,
        ) * self.action_mask[:, step].to(logits.dtype)

    @staticmethod
    @runtime_typed
    def _policy_cross_entropy(
        logits: Float[Tensor, "batch actions"],
        target: Float[Tensor, "batch actions"],
    ) -> Float[Tensor, "batch"]:
        if logits.shape != target.shape:
            raise ValueError("policy logits and targets must have the same shape")
        return -(target * functional.log_softmax(logits, dim=-1)).sum(dim=-1)

    @staticmethod
    @runtime_typed
    def _scalar_loss(
        logits: Float[Tensor, "batch support"],
        target: Float[Tensor, "batch"],
        *,
        support_min: int,
        support_max: int,
        epsilon: float = 0.001,
    ) -> Float[Tensor, "batch"]:
        if support_min >= support_max:
            raise ValueError("support_min must be less than support_max")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        expected_size = support_max - support_min + 1
        if logits.ndim != target.ndim + 1 or logits.shape[:-1] != target.shape:
            raise ValueError(
                "logits must have target.shape followed by a support axis"
            )
        if logits.shape[-1] != expected_size:
            raise ValueError(
                f"expected {expected_size} support logits, got {logits.shape[-1]}"
            )

        transformed = (
            target.sign() * (torch.sqrt(target.abs() + 1.0) - 1.0)
            + epsilon * target
        )
        transformed = transformed.clamp(support_min, support_max) - support_min
        lower = transformed.floor().long()
        upper = transformed.ceil().long()
        upper_weight = transformed - lower
        lower_weight = 1.0 - upper_weight

        # Cross entropy against an index is -log_softmax(logits)[index].
        # Compute the support-wide reduction once, then gather both adjacent
        # support atoms. Low-precision logits are reduced in FP32 for safety.
        reduction_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype
        )
        log_probabilities = functional.log_softmax(
            logits, dim=-1, dtype=reduction_dtype
        )
        lower_loss = -log_probabilities.gather(
            -1, lower.unsqueeze(-1)
        ).squeeze(-1)
        upper_loss = -log_probabilities.gather(
            -1, upper.unsqueeze(-1)
        ).squeeze(-1)
        return lower_weight * lower_loss + upper_weight * upper_loss

    @runtime_typed
    def value_prefix_targets(
        self, *, lstm_horizon: int
    ) -> Float[Tensor, "batch unroll"]:
        """Build cumulative reward targets, resetting at each LSTM horizon."""
        if self.rewards.shape != self.action_mask.shape:
            raise ValueError("rewards and action_mask must have the same shape")
        if self.rewards.ndim != 2:
            raise ValueError("rewards must have shape (batch, unroll_steps)")
        if lstm_horizon <= 0:
            raise ValueError("lstm_horizon must be positive")

        prefix = torch.zeros_like(self.rewards[:, 0])
        targets: list[Tensor] = []
        for step in range(self.rewards.shape[1]):
            prefix = prefix + self.rewards[:, step] * self.action_mask[
                :, step
            ].to(self.rewards.dtype)
            targets.append(prefix)
            if (step + 1) % lstm_horizon == 0:
                prefix = torch.zeros_like(prefix)
        return torch.stack(targets, dim=1)

    def to_reanalysis_device(
        self,
        device: torch.device | str,
    ) -> ReplayBatch:
        """Move only tensors required by target reanalysis to ``device``."""
        updates: dict[str, Tensor | None] = {
            "frames": self.frames.to(device),
            "policy_targets": self.policy_targets.to(device),
            "target_mask": self.target_mask.to(device),
            "value_targets": self.value_targets.to(device),
        }
        for name in (
            "value_bootstrap_frames",
            "value_bootstrap_values",
            "value_bootstrap_discounts",
            "value_bootstrap_mask",
        ):
            value = getattr(self, name)
            updates[name] = None if value is None else value.to(device)
        return replace(self, **updates)

    def pin_memory(self) -> ReplayBatch:
        """Copy CPU tensors into page-locked memory for asynchronous transfer."""
        return ReplayBatch(
            **{
                field.name: (
                    value.pin_memory() if isinstance(value, Tensor) else value
                )
                for field in fields(self)
                for value in (getattr(self, field.name),)
            }
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
        keep_indices_on_cpu: bool = False,
    ) -> ReplayBatch:
        """Move batch tensors to ``device``, optionally with async copies."""
        return ReplayBatch(
            **{
                field.name: (
                    value
                    if not isinstance(value, Tensor)
                    or (field.name == "indices" and keep_indices_on_cpu)
                    else value.to(device, non_blocking=non_blocking)
                )
                for field in fields(self)
                for value in (getattr(self, field.name),)
            }
        )



__all__ = ["ReplayBatch"]
