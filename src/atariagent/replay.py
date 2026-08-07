"""Small FIFO trajectory replay buffer with tensor batch sampling."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, fields

from einops import rearrange
import numpy as np
import torch
from torch import Tensor

from .agent import AtariObservation
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

    Observations are kept as ``uint8`` to avoid making a 256-sample Atari
    batch four times larger. Call :meth:`normalized_observations` immediately
    before network training. ``action_mask`` identifies real action/reward
    steps, while ``target_mask`` identifies states with stored MCTS targets.
    """

    observations: Tensor
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

    def normalized_observations(
        self, device: torch.device | str | None = None
    ) -> Tensor:
        """Return channel-first observations as float values in ``[0, 1]``."""
        return self.observations.to(device=device, dtype=torch.float32).div_(255.0)

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
        self._transition_count = 0
        self._action_space_size: int | None = None
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
        elif action_space_size != self._action_space_size:
            raise ValueError("all trajectories must use the same action space")

        evicted_trajectories = 0
        evicted_transitions = 0
        while self._transition_count + trajectory_length > self.max_transitions:
            evicted = self._trajectories.popleft()
            self._transition_count -= len(evicted)
            evicted_trajectories += 1
            evicted_transitions += len(evicted)

        self._trajectories.append(trajectory)
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

        A sample at position ``t`` contains observations ``t..t+K``, actions
        and rewards ``t..t+K-1``, and MCTS policy/value targets ``t..t+K``.
        Values beyond a trajectory boundary are padded and masked out.
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

        observations = torch.stack([sample["observations"] for sample in samples])
        return ReplayBatch(
            observations=observations,
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
        trajectory_length = len(trajectory)
        final_observation = trajectory.observations[-1]

        observation_tensors = []
        policy_targets = torch.zeros(
            unroll_steps + 1, self._action_space_size, dtype=torch.float32
        )
        root_values = torch.zeros(unroll_steps + 1, dtype=torch.float32)
        target_mask = torch.zeros(unroll_steps + 1, dtype=torch.bool)

        target_policy = trajectory.target_policy
        stored_root_values = trajectory.root_values
        for offset in range(unroll_steps + 1):
            state_position = start + offset
            observation = (
                trajectory.observations[state_position]
                if state_position <= trajectory_length
                else final_observation
            )
            observation_tensors.append(_uint8_observation_tensor(observation))
            if state_position < trajectory_length:
                policy_targets[offset] = torch.tensor(
                    target_policy[state_position], dtype=torch.float32
                )
                root_values[offset] = stored_root_values[state_position]
                target_mask[offset] = True

        actions = torch.zeros(unroll_steps, 1, dtype=torch.long)
        rewards = torch.zeros(unroll_steps, dtype=torch.float32)
        action_mask = torch.zeros(unroll_steps, dtype=torch.bool)

        for offset in range(unroll_steps):
            transition_position = start + offset
            if transition_position >= trajectory_length:
                break
            actions[offset, 0] = trajectory.actions[transition_position]
            rewards[offset] = trajectory.rewards[transition_position]
            action_mask[offset] = True

        return {
            "observations": torch.stack(observation_tensors),
            "actions": actions,
            "rewards": rewards,
            "policy_targets": policy_targets,
            "root_values": root_values,
            "action_mask": action_mask,
            "target_mask": target_mask,
        }


def _uint8_observation_tensor(observation: AtariObservation) -> Tensor:
    """Convert one stacked observation to channel-first uint8 storage."""
    frames = torch.as_tensor(np.asarray(observation), dtype=torch.uint8)
    if frames.ndim == 4:
        return rearrange(
            frames,
            "stack height width channels -> (stack channels) height width",
        )
    if frames.ndim == 3:
        return frames
    raise ValueError(
        "Atari observations must have shape (stack, H, W, C) or "
        "(channels, H, W)"
    )


__all__ = ["FIFOReplayBuffer", "ReplayAddResult", "ReplayBatch"]
