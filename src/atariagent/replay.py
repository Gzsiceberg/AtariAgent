"""Small FIFO trajectory replay buffer with tensor batch sampling."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, fields

import numpy as np
import torch
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
    action/reward steps. ``target_mask`` identifies valid state targets; a
    true terminal state has a valid zero-value target without an MCTS policy.
    """

    frames: Tensor
    actions: Tensor
    rewards: Tensor
    policy_targets: Tensor
    root_values: Tensor
    action_mask: Tensor
    target_mask: Tensor

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def unroll_steps(self) -> int:
        return self.actions.shape[1]

    @property
    def stack_size(self) -> int:
        return self.frames.shape[1] - self.unroll_steps

    def stacked_observations(self) -> Tensor:
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

    def normalized_observations(
        self, device: torch.device | str | None = None
    ) -> Tensor:
        """Return reconstructed state stacks as floats in ``[0, 1]``."""
        return self.stacked_observations().to(
            device=device, dtype=torch.float32
        ).div_(255.0)

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

    def sample(self, batch_size: int, *, unroll_steps: int = 5) -> ReplayBatch:
        """Uniformly sample unique starts and return padded tensor sequences.

        A sample at position ``t`` contains ``stack_size + K`` compact frames,
        actions/rewards ``t..t+K-1``, and policy/value targets ``t..t+K``.
        Consecutive nonterminal blocks are traversed when available; missing
        continuation is padded and masked out.
        """
        self._validate_sample_request(batch_size, unroll_steps)
        assert self._action_space_size is not None

        flat_indices = self._rng.choice(
            self._transition_count, size=batch_size, replace=False
        )
        locations = self._locations_for_indices(flat_indices)
        samples = [
            self._make_sample(trajectory, position, unroll_steps)
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
            action_mask=torch.stack(
                [sample["action_mask"] for sample in samples]
            ),
            target_mask=torch.stack(
                [sample["target_mask"] for sample in samples]
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
        self, batch_size: int, unroll_steps: int
    ) -> None:
        for value, name in (
            (batch_size, "batch_size"),
            (unroll_steps, "unroll_steps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
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
        target_mask = torch.zeros(unroll_steps + 1, dtype=torch.bool)
        actions = torch.zeros(unroll_steps, 1, dtype=torch.long)
        rewards = torch.zeros(unroll_steps, dtype=torch.float32)
        action_mask = torch.zeros(unroll_steps, dtype=torch.bool)

        block: GameTrajectory | None = trajectory
        position = start
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
                target_mask[offset + 1] = True
                block = None
                continue

            next_block = self._next_trajectory(block)
            if next_block is None:
                block = None
                continue

            block = next_block
            position = 0
            self._set_stored_target(
                policy_targets,
                root_values,
                target_mask,
                target_offset=offset + 1,
                trajectory=block,
                position=position,
            )

        while len(frame_sequence) < stack_size + unroll_steps:
            frame_sequence.append(frame_sequence[-1])

        return {
            "frames": torch.stack(frame_sequence),
            "actions": actions,
            "rewards": rewards,
            "policy_targets": policy_targets,
            "root_values": root_values,
            "action_mask": action_mask,
            "target_mask": target_mask,
        }

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
