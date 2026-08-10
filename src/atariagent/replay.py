"""Small FIFO trajectory replay buffer with tensor batch sampling."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, fields

from beartype import beartype
from jaxtyping import Bool, Float, Int, UInt8, jaxtyped
import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from .selfplay import GameTrajectory


@dataclass(frozen=True, slots=True)
class ReplayAddResult:
    """Summary of one replay insertion operation."""

    added_trajectories: int
    added_transitions: int
    evicted_trajectories: int
    evicted_transitions: int


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A padded EfficientZero-style unroll batch.

    ``frames`` contains the initial stack context followed by one new frame
    per unroll action. Call :meth:`normalized_observations` to reconstruct the
    overlapping state stacks for training. ``action_mask`` identifies real
    action/reward steps. ``target_mask`` identifies states with stored search
    targets. ``value_mask`` identifies states whose fixed-horizon return can
    be computed; a true terminal state has a valid zero-value target without
    an MCTS policy.
    """

    frames: UInt8[Tensor, "batch frames channels height width"]
    actions: Int[Tensor, "batch unroll 1"]
    rewards: Float[Tensor, "batch unroll"]
    policy_targets: Float[Tensor, "batch states actions"]
    root_values: Float[Tensor, "batch states"]
    value_targets: Float[Tensor, "batch states"]
    action_mask: Bool[Tensor, "batch unroll"]
    target_mask: Bool[Tensor, "batch states"]
    value_mask: Bool[Tensor, "batch states"]

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def unroll_steps(self) -> int:
        return self.actions.shape[1]

    @property
    def stack_size(self) -> int:
        return self.frames.shape[1] - self.unroll_steps

    @jaxtyped(typechecker=beartype)
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

    @jaxtyped(typechecker=beartype)
    def normalized_observations(
        self, device: torch.device | str | None = None
    ) -> Float[Tensor, "batch states stacked_channels height width"]:
        """Return reconstructed state stacks as floats in ``[0, 1]``."""
        return self.stacked_observations().to(
            device=device, dtype=torch.float32
        ).div_(255.0)

    @jaxtyped(typechecker=beartype)
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

    @jaxtyped(typechecker=beartype)
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
    @jaxtyped(typechecker=beartype)
    def _policy_cross_entropy(
        logits: Float[Tensor, "batch actions"],
        target: Float[Tensor, "batch actions"],
    ) -> Float[Tensor, "batch"]:
        if logits.shape != target.shape:
            raise ValueError("policy logits and targets must have the same shape")
        return -(target * functional.log_softmax(logits, dim=-1)).sum(dim=-1)

    @staticmethod
    @jaxtyped(typechecker=beartype)
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
        lower_loss = functional.cross_entropy(
            logits, lower, reduction="none"
        )
        upper_loss = functional.cross_entropy(
            logits, upper, reduction="none"
        )
        return lower_weight * lower_loss + upper_weight * upper_loss

    @jaxtyped(typechecker=beartype)
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

    def to(self, device: torch.device | str) -> ReplayBatch:
        """Move every batch tensor to ``device``."""
        return ReplayBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )


class FIFOReplayBuffer:
    """Store self-play trajectories and uniformly sample padded unrolls.

    Capacity is measured in transitions. Eviction removes the oldest complete
    trajectory blocks until the new block fits; priorities and reanalysis are
    intentionally outside the scope of this simple buffer.
    """

    def __init__(self, max_transitions: int, *, seed: int = 0) -> None:
        if isinstance(max_transitions, bool) or not isinstance(
            max_transitions, int
        ):
            raise TypeError("max_transitions must be an integer")
        if max_transitions <= 0:
            raise ValueError("max_transitions must be positive")

        self.max_transitions = max_transitions
        self._trajectories: deque[GameTrajectory] = deque()
        self._trajectory_by_key: dict[tuple[int, int, int], GameTrajectory] = {}
        self._transition_count = 0
        self._action_space_size: int | None = None
        self._stack_size: int | None = None
        self._frame_shape: tuple[int, ...] | None = None
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._transition_count

    @property
    def trajectory_count(self) -> int:
        return len(self._trajectories)

    @property
    def action_space_size(self) -> int | None:
        return self._action_space_size

    @property
    def utilization(self) -> float:
        return self._transition_count / self.max_transitions

    def add(self, trajectory: GameTrajectory) -> ReplayAddResult:
        """Append one trajectory and evict oldest complete blocks as needed."""
        trajectory_length = len(trajectory)
        if trajectory_length > self.max_transitions:
            raise ValueError(
                "trajectory length exceeds replay transition capacity"
            )

        action_space_size = self._validate_trajectory(trajectory)
        if self._action_space_size is None:
            self._action_space_size = action_space_size
            self._stack_size = trajectory.stack_size
            self._frame_shape = trajectory.frames[0].shape
        elif action_space_size != self._action_space_size:
            raise ValueError("all trajectories must use the same action space")
        elif trajectory.stack_size != self._stack_size:
            raise ValueError("all trajectories must use the same stack size")
        elif trajectory.frames[0].shape != self._frame_shape:
            raise ValueError("all trajectories must use the same frame shape")

        key = self._trajectory_key(trajectory)
        if key in self._trajectory_by_key:
            raise ValueError("trajectory identity already exists in replay")

        evicted_trajectories = 0
        evicted_transitions = 0
        while self._transition_count + trajectory_length > self.max_transitions:
            evicted = self._trajectories.popleft()
            del self._trajectory_by_key[self._trajectory_key(evicted)]
            self._transition_count -= len(evicted)
            evicted_trajectories += 1
            evicted_transitions += len(evicted)

        self._trajectories.append(trajectory)
        self._trajectory_by_key[key] = trajectory
        self._transition_count += trajectory_length
        return ReplayAddResult(
            added_trajectories=1,
            added_transitions=trajectory_length,
            evicted_trajectories=evicted_trajectories,
            evicted_transitions=evicted_transitions,
        )

    def extend(self, trajectories: Iterable[GameTrajectory]) -> ReplayAddResult:
        """Append trajectories in order and combine their insertion statistics."""
        added_trajectories = 0
        added_transitions = 0
        evicted_trajectories = 0
        evicted_transitions = 0
        for trajectory in trajectories:
            result = self.add(trajectory)
            added_trajectories += result.added_trajectories
            added_transitions += result.added_transitions
            evicted_trajectories += result.evicted_trajectories
            evicted_transitions += result.evicted_transitions
        return ReplayAddResult(
            added_trajectories=added_trajectories,
            added_transitions=added_transitions,
            evicted_trajectories=evicted_trajectories,
            evicted_transitions=evicted_transitions,
        )

    def sample(
        self,
        batch_size: int,
        *,
        unroll_steps: int = 5,
        td_steps: int = 5,
        discount: float = 0.997,
    ) -> ReplayBatch:
        """Uniformly sample unique starts and return padded tensor sequences.

        A sample at position ``t`` contains ``stack_size + K`` compact frames,
        actions/rewards ``t..t+K-1``, and policy/value targets ``t..t+K``.
        Value targets use a fixed ``td_steps`` discounted reward return,
        bootstrapped from the stored MCTS root value. Consecutive nonterminal
        blocks are traversed when available; missing continuation is padded
        and masked out.
        """
        self._validate_sample_request(
            batch_size, unroll_steps, td_steps, discount
        )
        assert self._action_space_size is not None

        flat_indices = self._rng.choice(
            self._transition_count, size=batch_size, replace=False
        )
        locations = self._locations_for_indices(flat_indices)
        samples = [
            self._make_sample(
                trajectory,
                position,
                unroll_steps,
                td_steps,
                discount,
            )
            for trajectory, position in locations
        ]

        frames = torch.stack([sample["frames"] for sample in samples])
        return ReplayBatch(
            frames=frames,
            actions=torch.stack([sample["actions"] for sample in samples]),
            rewards=torch.stack([sample["rewards"] for sample in samples]),
            policy_targets=torch.stack(
                [sample["policy_targets"] for sample in samples]
            ),
            root_values=torch.stack(
                [sample["root_values"] for sample in samples]
            ),
            value_targets=torch.stack(
                [sample["value_targets"] for sample in samples]
            ),
            action_mask=torch.stack(
                [sample["action_mask"] for sample in samples]
            ),
            target_mask=torch.stack(
                [sample["target_mask"] for sample in samples]
            ),
            value_mask=torch.stack(
                [sample["value_mask"] for sample in samples]
            ),
        )

    def _validate_trajectory(self, trajectory: GameTrajectory) -> int:
        action_space_sizes = {
            len(distribution) for distribution in trajectory.target_policy
        }
        if len(action_space_sizes) != 1:
            raise ValueError(
                "every policy target in a trajectory must have the same size"
            )
        action_space_size = action_space_sizes.pop()
        if action_space_size <= 0:
            raise ValueError("policy targets must not be empty")
        return action_space_size

    def _validate_sample_request(
        self,
        batch_size: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> None:
        for value, name in (
            (batch_size, "batch_size"),
            (unroll_steps, "unroll_steps"),
            (td_steps, "td_steps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")
        if batch_size > self._transition_count:
            raise ValueError(
                f"cannot sample {batch_size} unique transitions from "
                f"a replay buffer containing {self._transition_count}"
            )

    def _locations_for_indices(
        self, flat_indices: np.ndarray
    ) -> list[tuple[GameTrajectory, int]]:
        ordered = sorted(
            enumerate(int(index) for index in flat_indices), key=lambda item: item[1]
        )
        resolved: list[tuple[GameTrajectory, int] | None] = [None] * len(ordered)
        trajectory_offset = 0
        ordered_index = 0

        for trajectory in self._trajectories:
            trajectory_end = trajectory_offset + len(trajectory)
            while (
                ordered_index < len(ordered)
                and ordered[ordered_index][1] < trajectory_end
            ):
                original_position, flat_index = ordered[ordered_index]
                resolved[original_position] = (
                    trajectory,
                    flat_index - trajectory_offset,
                )
                ordered_index += 1
            trajectory_offset = trajectory_end
            if ordered_index == len(ordered):
                break

        if any(location is None for location in resolved):
            raise RuntimeError("failed to resolve sampled replay indices")
        return [location for location in resolved if location is not None]

    def _make_sample(
        self,
        trajectory: GameTrajectory,
        start: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> dict[str, Tensor]:
        assert self._action_space_size is not None
        stack_size = trajectory.stack_size
        frame_sequence = [
            torch.as_tensor(frame, dtype=torch.uint8)
            for frame in trajectory.frames[start : start + stack_size]
        ]

        policy_targets = torch.zeros(
            unroll_steps + 1, self._action_space_size, dtype=torch.float32
        )
        root_values = torch.zeros(unroll_steps + 1, dtype=torch.float32)
        value_targets = torch.zeros(unroll_steps + 1, dtype=torch.float32)
        target_mask = torch.zeros(unroll_steps + 1, dtype=torch.bool)
        value_mask = torch.zeros(unroll_steps + 1, dtype=torch.bool)
        actions = torch.zeros(unroll_steps, 1, dtype=torch.long)
        rewards = torch.zeros(unroll_steps, dtype=torch.float32)
        action_mask = torch.zeros(unroll_steps, dtype=torch.bool)

        block: GameTrajectory | None = trajectory
        position = start
        target_locations: list[tuple[GameTrajectory, int] | None] = [
            (trajectory, start)
        ]
        terminal_target_offsets: set[int] = set()
        self._set_stored_target(
            policy_targets,
            root_values,
            target_mask,
            target_offset=0,
            trajectory=trajectory,
            position=start,
        )

        for offset in range(unroll_steps):
            if block is None or position >= len(block):
                break

            actions[offset, 0] = block.actions[position]
            rewards[offset] = block.rewards[position]
            action_mask[offset] = True
            frame_sequence.append(
                torch.as_tensor(
                    block.frames[position + block.stack_size],
                    dtype=torch.uint8,
                )
            )
            position += 1

            if position < len(block):
                target_locations.append((block, position))
                self._set_stored_target(
                    policy_targets,
                    root_values,
                    target_mask,
                    target_offset=offset + 1,
                    trajectory=block,
                    position=position,
                )
                continue

            if block.terminated:
                # No policy exists at a true terminal state, but value zero is
                # valid supervision for the recurrent state reached here.
                target_locations.append(None)
                terminal_target_offsets.add(offset + 1)
                target_mask[offset + 1] = True
                break

            next_block = self._next_trajectory(block)
            if next_block is None:
                target_locations.append(None)
                break

            block = next_block
            position = 0
            target_locations.append((block, position))
            self._set_stored_target(
                policy_targets,
                root_values,
                target_mask,
                target_offset=offset + 1,
                trajectory=block,
                position=position,
            )

        while len(target_locations) < unroll_steps + 1:
            target_locations.append(None)

        for target_offset, location in enumerate(target_locations):
            if target_offset in terminal_target_offsets:
                value_mask[target_offset] = True
                continue
            if location is None:
                continue
            value_target, valid = self._n_step_value_target(
                *location,
                td_steps=td_steps,
                discount=discount,
            )
            if valid:
                value_targets[target_offset] = value_target
                value_mask[target_offset] = True

        while len(frame_sequence) < stack_size + unroll_steps:
            frame_sequence.append(frame_sequence[-1])

        return {
            "frames": torch.stack(frame_sequence),
            "actions": actions,
            "rewards": rewards,
            "policy_targets": policy_targets,
            "root_values": root_values,
            "value_targets": value_targets,
            "action_mask": action_mask,
            "target_mask": target_mask,
            "value_mask": value_mask,
        }

    def _n_step_value_target(
        self,
        trajectory: GameTrajectory,
        position: int,
        *,
        td_steps: int,
        discount: float,
    ) -> tuple[float, bool]:
        """Return a fixed-horizon reward return and stored-root bootstrap."""
        value = 0.0
        discount_power = 1.0
        block = trajectory

        for _ in range(td_steps):
            value += discount_power * block.rewards[position]
            discount_power *= discount
            position += 1

            if position < len(block):
                continue
            if block.terminated:
                return value, True

            next_block = self._next_trajectory(block)
            if next_block is None:
                return 0.0, False
            block = next_block
            position = 0

        bootstrap_value = block.search_results[position].root_value
        return value + discount_power * bootstrap_value, True

    def _set_stored_target(
        self,
        policy_targets: Tensor,
        root_values: Tensor,
        target_mask: Tensor,
        *,
        target_offset: int,
        trajectory: GameTrajectory,
        position: int,
    ) -> None:
        search_result = trajectory.search_results[position]
        visit_counts = torch.tensor(
            search_result.visit_counts, dtype=torch.float32
        )
        policy_targets[target_offset] = visit_counts / visit_counts.sum()
        root_values[target_offset] = search_result.root_value
        target_mask[target_offset] = True

    def _next_trajectory(
        self, trajectory: GameTrajectory
    ) -> GameTrajectory | None:
        if trajectory.terminated or trajectory.truncated:
            return None
        return self._trajectory_by_key.get(
            (
                trajectory.environment_index,
                trajectory.episode_id,
                trajectory.block_id + 1,
            )
        )

    @staticmethod
    def _trajectory_key(trajectory: GameTrajectory) -> tuple[int, int, int]:
        return (
            trajectory.environment_index,
            trajectory.episode_id,
            trajectory.block_id,
        )


__all__ = ["FIFOReplayBuffer", "ReplayAddResult", "ReplayBatch"]
