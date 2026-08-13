"""FIFO trajectory replay buffer with prioritized tensor-batch sampling."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .replay_batch import ReplayBatch
from .selfplay import GameTrajectory


@dataclass(frozen=True, slots=True)
class ReplayAddResult:
    """Summary of one replay insertion operation."""

    added_trajectories: int
    added_transitions: int
    evicted_trajectories: int
    evicted_transitions: int


@dataclass(frozen=True, slots=True)
class _ValueTarget:
    """Replay value target and optional fixed-horizon bootstrap metadata."""

    value: float
    valid: bool
    bootstrap_location: tuple[GameTrajectory, int] | None = None
    bootstrap_value: float = 0.0
    bootstrap_discount: float = 0.0


class FIFOReplayBuffer:
    """Store trajectories with FIFO eviction and prioritized sampling.

    New sampleable transitions receive the current maximum priority; trailing
    lookahead transitions remain attached to their trajectory as local target
    context. Sampled transitions carry normalized importance weights and stable
    IDs used to update their priorities after training. Reanalysis remains
    outside this buffer.
    """

    def __init__(
        self,
        max_transitions: int,
        *,
        seed: int = 0,
        priority_alpha: float = 0.6,
    ) -> None:
        if isinstance(max_transitions, bool) or not isinstance(
            max_transitions, int
        ):
            raise TypeError("max_transitions must be an integer")
        if max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        if not np.isfinite(priority_alpha) or priority_alpha < 0.0:
            raise ValueError("priority_alpha must be finite and non-negative")

        self.max_transitions = max_transitions
        self.priority_alpha = float(priority_alpha)
        self._trajectories: deque[GameTrajectory] = deque()
        self._transition_ids = np.empty(0, dtype=np.int64)
        self._priorities = np.empty(0, dtype=np.float64)
        self._next_transition_id = 0
        self._trajectory_keys: set[tuple[int, int, int]] = set()
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

    @property
    def priorities(self) -> np.ndarray:
        """Return priorities in the same flat order used for sampling."""
        return self._priorities.copy()

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
        if key in self._trajectory_keys:
            raise ValueError("trajectory identity already exists in replay")

        evicted_trajectories = 0
        evicted_transitions = 0
        while self._transition_count + trajectory_length > self.max_transitions:
            evicted = self._trajectories.popleft()
            self._trajectory_keys.remove(self._trajectory_key(evicted))
            self._transition_count -= len(evicted)
            evicted_trajectories += 1
            evicted_transitions += len(evicted)

        if evicted_transitions:
            self._transition_ids = self._transition_ids[evicted_transitions:]
            self._priorities = self._priorities[evicted_transitions:]

        maximum_priority = (
            float(self._priorities.max()) if self._priorities.size else 1.0
        )
        transition_ids = np.arange(
            self._next_transition_id,
            self._next_transition_id + trajectory_length,
            dtype=np.int64,
        )
        self._next_transition_id += trajectory_length
        self._transition_ids = np.concatenate(
            (self._transition_ids, transition_ids)
        )
        self._priorities = np.concatenate(
            (
                self._priorities,
                np.full(trajectory_length, maximum_priority, dtype=np.float64),
            )
        )

        self._trajectories.append(trajectory)
        self._trajectory_keys.add(key)
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
        priority_beta: float = 0.4,
        include_value_bootstraps: bool = True,
    ) -> ReplayBatch:
        """Prioritize unique starts and slice their local lookahead context."""
        self._validate_sample_request(
            batch_size, unroll_steps, td_steps, discount, priority_beta
        )
        self._validate_lookahead(unroll_steps, td_steps)
        if not isinstance(include_value_bootstraps, bool):
            raise TypeError("include_value_bootstraps must be a boolean")
        assert self._action_space_size is not None

        locations, transition_ids, importance_weights = self._sample_context(
            batch_size, priority_beta
        )
        arrays = self._allocate_batch_arrays(
            batch_size,
            unroll_steps,
            include_value_bootstraps=include_value_bootstraps,
        )
        self._fill_batch_arrays(
            arrays,
            locations,
            transition_ids=transition_ids,
            importance_weights=importance_weights,
            unroll_steps=unroll_steps,
            td_steps=td_steps,
            discount=discount,
        )
        return self._batch_from_arrays(arrays)

    def update_priorities(
        self,
        indices: Tensor | np.ndarray | Iterable[int],
        priorities: Tensor | np.ndarray | Iterable[float],
    ) -> None:
        """Update sampled transition priorities by stable transition ID."""
        if isinstance(indices, Tensor):
            index_array = indices.detach().cpu().numpy()
        else:
            index_array = np.asarray(
                tuple(indices) if not isinstance(indices, np.ndarray) else indices
            )
        if isinstance(priorities, Tensor):
            priority_array = priorities.detach().cpu().numpy()
        else:
            priority_array = np.asarray(
                tuple(priorities)
                if not isinstance(priorities, np.ndarray)
                else priorities,
                dtype=np.float64,
            )
        index_array = np.asarray(index_array).reshape(-1)
        priority_array = np.asarray(priority_array, dtype=np.float64).reshape(-1)
        if index_array.shape != priority_array.shape:
            raise ValueError("indices and priorities must have the same shape")
        if not np.issubdtype(index_array.dtype, np.integer):
            raise TypeError("indices must be integers")
        if not np.all(np.isfinite(priority_array)) or np.any(
            priority_array <= 0.0
        ):
            raise ValueError("priorities must be finite and positive")
        if index_array.size == 0:
            return
        if self._transition_ids.size == 0:
            raise KeyError("sampled transitions are no longer in replay")
        offsets = index_array.astype(np.int64) - self._transition_ids[0]
        valid = (offsets >= 0) & (offsets < self._transition_ids.size)
        if np.any(valid):
            safe_offsets = np.clip(offsets, 0, self._transition_ids.size - 1)
            valid &= self._transition_ids[safe_offsets] == index_array
        if not np.all(valid):
            stale_id = int(index_array[~valid][0])
            raise KeyError(f"transition ID {stale_id} is no longer in replay")
        self._priorities[offsets] = priority_array

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
        priority_beta: float,
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
        if not np.isfinite(priority_beta) or not 0.0 <= priority_beta <= 1.0:
            raise ValueError("priority_beta must be finite and in [0, 1]")
        if batch_size > self._transition_count:
            raise ValueError(
                f"cannot sample {batch_size} unique transitions from "
                f"a replay buffer containing {self._transition_count}"
            )

    def _validate_lookahead(self, unroll_steps: int, td_steps: int) -> None:
        """Require complete nonterminal tails to cover both target horizons."""
        for trajectory in self._trajectories:
            trajectory.validate_lookahead(unroll_steps, td_steps)

    def _sample_context(
        self, batch_size: int, priority_beta: float
    ) -> tuple[
        list[tuple[GameTrajectory, int]],
        np.ndarray,
        np.ndarray,
    ]:
        probabilities = self._priorities**self.priority_alpha
        probabilities /= probabilities.sum()
        flat_indices = self._rng.choice(
            self._transition_count,
            size=batch_size,
            replace=False,
            p=probabilities,
        )
        sampled_probabilities = probabilities[flat_indices]
        importance_weights = (
            self._transition_count * sampled_probabilities
        ) ** (-priority_beta)
        importance_weights /= importance_weights.max()
        return (
            self._locations_for_indices(flat_indices),
            self._transition_ids[flat_indices],
            importance_weights,
        )

    def _batch_array_specs(
        self,
        batch_size: int,
        unroll_steps: int,
        *,
        include_value_bootstraps: bool,
    ) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        assert self._action_space_size is not None
        assert self._stack_size is not None
        assert self._frame_shape is not None
        states = unroll_steps + 1
        specs = {
            "frames": (
                (
                    batch_size,
                    self._stack_size + unroll_steps,
                    *self._frame_shape,
                ),
                np.dtype(np.uint8),
            ),
            "actions": (
                (batch_size, unroll_steps, 1),
                np.dtype(np.int64),
            ),
            "rewards": ((batch_size, unroll_steps), np.dtype(np.float32)),
            "policy_targets": (
                (batch_size, states, self._action_space_size),
                np.dtype(np.float32),
            ),
            "value_targets": ((batch_size, states), np.dtype(np.float32)),
            "action_mask": ((batch_size, unroll_steps), np.dtype(np.bool_)),
            "policy_mask": ((batch_size, states), np.dtype(np.bool_)),
            "value_mask": ((batch_size, states), np.dtype(np.bool_)),
            "value_bootstrap_frames": (
                (
                    batch_size,
                    self._stack_size + unroll_steps,
                    *self._frame_shape,
                ),
                np.dtype(np.uint8),
            ),
            "value_bootstrap_values": (
                (batch_size, states),
                np.dtype(np.float32),
            ),
            "value_bootstrap_discounts": (
                (batch_size, states),
                np.dtype(np.float32),
            ),
            "value_bootstrap_mask": (
                (batch_size, states),
                np.dtype(np.bool_),
            ),
            "indices": ((batch_size,), np.dtype(np.int64)),
            "importance_weights": ((batch_size,), np.dtype(np.float32)),
        }
        if not include_value_bootstraps:
            for name in (
                "value_bootstrap_frames",
                "value_bootstrap_values",
                "value_bootstrap_discounts",
                "value_bootstrap_mask",
            ):
                del specs[name]
        return specs

    def _allocate_batch_arrays(
        self,
        batch_size: int,
        unroll_steps: int,
        *,
        include_value_bootstraps: bool,
    ) -> dict[str, np.ndarray]:
        return {
            name: np.empty(shape, dtype=dtype)
            for name, (shape, dtype) in self._batch_array_specs(
                batch_size,
                unroll_steps,
                include_value_bootstraps=include_value_bootstraps,
            ).items()
        }

    @staticmethod
    def _batch_from_arrays(arrays: Mapping[str, np.ndarray]) -> ReplayBatch:
        def optional_tensor(name: str) -> Tensor | None:
            array = arrays.get(name)
            return None if array is None else torch.from_numpy(array)

        return ReplayBatch(
            frames=torch.from_numpy(arrays["frames"]),
            actions=torch.from_numpy(arrays["actions"]),
            rewards=torch.from_numpy(arrays["rewards"]),
            policy_targets=torch.from_numpy(arrays["policy_targets"]),
            value_targets=torch.from_numpy(arrays["value_targets"]),
            action_mask=torch.from_numpy(arrays["action_mask"]),
            policy_mask=torch.from_numpy(arrays["policy_mask"]),
            value_mask=torch.from_numpy(arrays["value_mask"]),
            indices=torch.from_numpy(arrays["indices"]),
            importance_weights=torch.from_numpy(arrays["importance_weights"]),
            value_bootstrap_frames=optional_tensor("value_bootstrap_frames"),
            value_bootstrap_values=optional_tensor("value_bootstrap_values"),
            value_bootstrap_discounts=optional_tensor(
                "value_bootstrap_discounts"
            ),
            value_bootstrap_mask=optional_tensor("value_bootstrap_mask"),
        )

    def _locations_for_indices(
        self, flat_indices: np.ndarray
    ) -> list[tuple[GameTrajectory, int]]:
        """Resolve flat replay offsets with vectorized cumulative boundaries."""
        trajectories = tuple(self._trajectories)
        lengths = np.fromiter(
            (len(trajectory) for trajectory in trajectories),
            dtype=np.int64,
            count=len(trajectories),
        )
        ends = np.cumsum(lengths)
        trajectory_indices = np.searchsorted(ends, flat_indices, side="right")
        starts = ends - lengths
        positions = flat_indices - starts[trajectory_indices]
        return [
            (trajectories[int(trajectory_index)], int(position))
            for trajectory_index, position in zip(
                trajectory_indices, positions, strict=True
            )
        ]

    def _fill_batch_arrays(
        self,
        arrays: Mapping[str, np.ndarray],
        locations: list[tuple[GameTrajectory, int]],
        *,
        transition_ids: np.ndarray,
        importance_weights: np.ndarray,
        unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> None:
        """Fill a complete batch in reusable NumPy arrays."""
        assert self._stack_size is not None

        stack_size = self._stack_size
        state_count = unroll_steps + 1
        frames = arrays["frames"]
        actions = arrays["actions"]
        rewards = arrays["rewards"]
        policy_targets = arrays["policy_targets"]
        value_targets = arrays["value_targets"]
        action_mask = arrays["action_mask"]
        policy_mask = arrays["policy_mask"]
        value_mask = arrays["value_mask"]
        value_bootstrap_frames = arrays.get("value_bootstrap_frames")
        value_bootstrap_values = arrays.get("value_bootstrap_values")
        value_bootstrap_discounts = arrays.get("value_bootstrap_discounts")
        value_bootstrap_mask = arrays.get("value_bootstrap_mask")
        arrays_to_clear = (
            actions,
            rewards,
            policy_targets,
            value_targets,
            action_mask,
            policy_mask,
            value_mask,
        )
        if value_bootstrap_frames is not None:
            assert value_bootstrap_values is not None
            assert value_bootstrap_discounts is not None
            assert value_bootstrap_mask is not None
            arrays_to_clear += (
                value_bootstrap_frames,
                value_bootstrap_values,
                value_bootstrap_discounts,
                value_bootstrap_mask,
            )
        for array in arrays_to_clear:
            array.fill(0)

        for batch_index, (trajectory, start) in enumerate(locations):
            stored_count = len(trajectory) + trajectory.lookahead_steps
            action_count = min(unroll_steps, stored_count - start)
            frame_count = stack_size + action_count
            frames[batch_index, :frame_count] = np.asarray(
                trajectory.frames[start : start + frame_count],
                dtype=np.uint8,
            )
            if frame_count < stack_size + unroll_steps:
                frames[batch_index, frame_count:] = frames[
                    batch_index, frame_count - 1
                ]

            action_end = start + action_count
            actions[batch_index, :action_count, 0] = np.asarray(
                trajectory.actions[start:action_end], dtype=np.int64
            )
            rewards[batch_index, :action_count] = np.asarray(
                trajectory.rewards[start:action_end], dtype=np.float32
            )
            action_mask[batch_index, :action_count] = True

            policy_count = min(state_count, stored_count - start)
            policy_end = start + policy_count
            visit_counts = np.asarray(
                tuple(
                    result.visit_counts
                    for result in trajectory.search_results[start:policy_end]
                ),
                dtype=np.float32,
            )
            policy_targets[batch_index, :policy_count] = visit_counts / (
                visit_counts.sum(axis=1, keepdims=True)
            )
            policy_mask[batch_index, :policy_count] = True

            value_target_items = self._n_step_value_targets(
                trajectory,
                start,
                target_count=state_count,
                td_steps=td_steps,
                discount=discount,
            )
            for target_offset, target in enumerate(value_target_items):
                if target.valid:
                    value_targets[batch_index, target_offset] = target.value
                    value_mask[batch_index, target_offset] = True
                if (
                    target.bootstrap_location is not None
                    and value_bootstrap_frames is not None
                ):
                    assert value_bootstrap_values is not None
                    assert value_bootstrap_discounts is not None
                    assert value_bootstrap_mask is not None
                    bootstrap_block, bootstrap_position = (
                        target.bootstrap_location
                    )
                    value_bootstrap_frames[
                        batch_index,
                        target_offset : target_offset + stack_size,
                    ] = np.asarray(
                        bootstrap_block.frames[
                            bootstrap_position : bootstrap_position + stack_size
                        ],
                        dtype=np.uint8,
                    )
                    value_bootstrap_values[
                        batch_index, target_offset
                    ] = target.bootstrap_value
                    value_bootstrap_discounts[
                        batch_index, target_offset
                    ] = target.bootstrap_discount
                    value_bootstrap_mask[batch_index, target_offset] = True

        arrays["indices"][:] = transition_ids
        arrays["importance_weights"][:] = importance_weights

    def _n_step_value_targets(
        self,
        trajectory: GameTrajectory,
        position: int,
        *,
        target_count: int,
        td_steps: int,
        discount: float,
    ) -> list[_ValueTarget]:
        """Build consecutive n-step targets from one trajectory in linear time.

        The trajectory's duplicated lookahead tail supplies rewards and root
        values beyond its sampleable range, so no next-block lookup is needed.
        Discounted suffix sums produce each reward return in constant time.
        """
        maximum_rewards = target_count - 1 + td_steps
        stored_count = len(trajectory) + trajectory.lookahead_steps
        available_transitions = stored_count - position
        reward_count = min(maximum_rewards, available_transitions)
        rewards = np.asarray(
            trajectory.rewards[position : position + reward_count],
            dtype=np.float64,
        )
        ended_at_terminal = (
            trajectory.terminated and reward_count == available_transitions
        )

        discount_powers = discount ** np.arange(td_steps + 1)
        discounted_suffix = np.zeros(reward_count + 1, dtype=np.float64)
        for reward_offset in range(reward_count - 1, -1, -1):
            discounted_suffix[reward_offset] = (
                rewards[reward_offset]
                + discount * discounted_suffix[reward_offset + 1]
            )

        targets: list[_ValueTarget] = []
        terminal_state = available_transitions if ended_at_terminal else -1
        for target_offset in range(target_count):
            if target_offset == terminal_state:
                targets.append(_ValueTarget(value=0.0, valid=True))
                continue
            if target_offset >= available_transitions:
                targets.append(_ValueTarget(value=0.0, valid=False))
                continue

            bootstrap_offset = target_offset + td_steps
            reward_end = min(bootstrap_offset, reward_count)
            reward_horizon = reward_end - target_offset
            reward_return = float(
                discounted_suffix[target_offset]
                - discount_powers[reward_horizon]
                * discounted_suffix[reward_end]
            )

            if bootstrap_offset > reward_count:
                targets.append(
                    _ValueTarget(
                        value=(reward_return if ended_at_terminal else 0.0),
                        valid=ended_at_terminal,
                    )
                )
                continue
            if bootstrap_offset == terminal_state:
                targets.append(_ValueTarget(value=reward_return, valid=True))
                continue

            bootstrap_position = position + bootstrap_offset
            if bootstrap_position >= stored_count:
                targets.append(_ValueTarget(value=0.0, valid=False))
                continue

            bootstrap_value = trajectory.search_results[
                bootstrap_position
            ].root_value
            bootstrap_discount = float(discount_powers[td_steps])
            targets.append(
                _ValueTarget(
                    value=(
                        reward_return + bootstrap_discount * bootstrap_value
                    ),
                    valid=True,
                    bootstrap_location=(trajectory, bootstrap_position),
                    bootstrap_value=bootstrap_value,
                    bootstrap_discount=bootstrap_discount,
                )
            )
        return targets

    @staticmethod
    def _trajectory_key(trajectory: GameTrajectory) -> tuple[int, int, int]:
        return (
            trajectory.environment_index,
            trajectory.episode_id,
            trajectory.block_id,
        )


__all__ = ["FIFOReplayBuffer", "ReplayAddResult", "ReplayBatch"]
