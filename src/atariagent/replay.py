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
class _StoredTrajectory:
    """Contiguous replay data and targets prepared during insertion."""

    environment_index: int
    episode_id: int
    block_id: int
    stack_size: int
    sampleable_transition_count: int
    lookahead_steps: int
    terminated: bool
    truncated: bool
    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    policy_targets: np.ndarray
    root_values: np.ndarray
    predicted_values: np.ndarray
    initial_priority: float
    value_targets: np.ndarray
    value_valid_mask: np.ndarray

    def __len__(self) -> int:
        return self.sampleable_transition_count

    @property
    def stored_transition_count(self) -> int:
        return int(self.actions.shape[0])


class FIFOReplayBuffer:
    """Store prepared trajectories with FIFO prioritized sampling.

    Target horizons and discount are fixed for the buffer lifetime. New
    sampleable transitions receive the current maximum priority; trailing
    lookahead transitions remain local target context and are not replay starts.
    Reanalysis remains outside this buffer.
    """

    def __init__(
        self,
        max_transitions: int,
        *,
        unroll_steps: int = 5,
        td_steps: int = 5,
        discount: float = 0.997,
        priority_epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if isinstance(max_transitions, bool) or not isinstance(max_transitions, int):
            raise TypeError("max_transitions must be an integer")
        if max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        for value, name in (
            (unroll_steps, "unroll_steps"),
            (td_steps, "td_steps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")
        if not np.isfinite(priority_epsilon) or priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be finite and positive")

        self.max_transitions = max_transitions
        self._unroll_steps = unroll_steps
        self._td_steps = td_steps
        self._discount = float(discount)
        self._priority_epsilon = float(priority_epsilon)
        self._reward_discounts = self.discount ** np.arange(
            self.td_steps, dtype=np.float64
        )
        self._bootstrap_discount = float(self.discount**self.td_steps)
        self._trajectories: deque[_StoredTrajectory] = deque()
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
        """Return the number of stored sampleable transitions."""
        return self._transition_count

    @property
    def trajectory_count(self) -> int:
        return len(self._trajectories)

    @property
    def unroll_steps(self) -> int:
        return self._unroll_steps

    @property
    def td_steps(self) -> int:
        return self._td_steps

    @property
    def discount(self) -> float:
        return self._discount

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
        """Prepare one trajectory, then append it with FIFO eviction."""
        trajectory_length = len(trajectory)
        if trajectory_length > self.max_transitions:
            raise ValueError("trajectory length exceeds replay transition capacity")

        key = self._trajectory_key(trajectory)
        if key in self._trajectory_keys:
            raise ValueError("trajectory identity already exists in replay")

        # Preparation and all validation happen before replay state is mutated.
        trajectory.validate_lookahead(self.unroll_steps, self.td_steps)
        stored = self._prepare_trajectory(trajectory)
        action_space_size = int(stored.policy_targets.shape[1])
        frame_shape = tuple(stored.frames.shape[1:])
        if self._action_space_size is not None:
            if action_space_size != self._action_space_size:
                raise ValueError("all trajectories must use the same action space")
            if stored.stack_size != self._stack_size:
                raise ValueError("all trajectories must use the same stack size")
            if frame_shape != self._frame_shape:
                raise ValueError("all trajectories must use the same frame shape")

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
        maximum_priority = max(maximum_priority, stored.initial_priority)
        transition_ids = np.arange(
            self._next_transition_id,
            self._next_transition_id + trajectory_length,
            dtype=np.int64,
        )
        self._next_transition_id += trajectory_length
        self._transition_ids = np.concatenate((self._transition_ids, transition_ids))
        self._priorities = np.concatenate(
            (
                self._priorities,
                np.full(trajectory_length, maximum_priority, dtype=np.float64),
            )
        )

        if self._action_space_size is None:
            self._action_space_size = action_space_size
            self._stack_size = stored.stack_size
            self._frame_shape = frame_shape
        self._trajectories.append(stored)
        self._trajectory_keys.add(key)
        self._transition_count += trajectory_length
        return ReplayAddResult(
            added_trajectories=1,
            added_transitions=trajectory_length,
            evicted_trajectories=evicted_trajectories,
            evicted_transitions=evicted_transitions,
        )

    def extend(self, trajectories: Iterable[GameTrajectory]) -> ReplayAddResult:
        """Append trajectories in order and combine insertion statistics."""
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
        include_value_bootstraps: bool = True,
        pin_memory: bool = False,
    ) -> ReplayBatch:
        """Prioritize unique starts and copy their prepared local context."""
        self._validate_sample_request(batch_size)
        if not isinstance(include_value_bootstraps, bool):
            raise TypeError("include_value_bootstraps must be a boolean")
        if not isinstance(pin_memory, bool):
            raise TypeError("pin_memory must be a boolean")
        assert self._action_space_size is not None

        locations, transition_ids, importance_weights = self._sample_context(batch_size)
        arrays = self._allocate_batch_arrays(
            batch_size,
            include_value_bootstraps=include_value_bootstraps,
            pin_memory=pin_memory,
        )
        self._fill_batch_arrays(
            arrays,
            locations,
            transition_ids=transition_ids,
            importance_weights=importance_weights,
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
        if not np.all(np.isfinite(priority_array)) or np.any(priority_array <= 0.0):
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

    def _prepare_trajectory(self, trajectory: GameTrajectory) -> _StoredTrajectory:
        """Convert one immutable self-play block to contiguous replay data."""
        action_space_sizes = {
            len(result.visit_counts) for result in trajectory.search_results
        }
        if len(action_space_sizes) != 1:
            raise ValueError(
                "every policy target in a trajectory must have the same size"
            )
        action_space_size = action_space_sizes.pop()
        if action_space_size <= 0:
            raise ValueError("policy targets must not be empty")

        frames = np.ascontiguousarray(np.asarray(trajectory.frames, dtype=np.uint8))
        actions = np.ascontiguousarray(np.asarray(trajectory.actions, dtype=np.int64))
        rewards64 = np.asarray(trajectory.rewards, dtype=np.float64)
        rewards = np.ascontiguousarray(rewards64, dtype=np.float32)
        visit_counts = np.asarray(
            [result.visit_counts for result in trajectory.search_results],
            dtype=np.float32,
        )
        if visit_counts.shape != (
            trajectory.stored_transition_count,
            action_space_size,
        ):
            raise ValueError("policy targets have an invalid shape")
        visit_totals = visit_counts.sum(axis=1, keepdims=True)
        if np.any(visit_totals <= 0.0):
            raise ValueError("MCTS search result must contain visited actions")
        policy_targets = np.ascontiguousarray(visit_counts / visit_totals)
        root_values64 = np.fromiter(
            (result.root_value for result in trajectory.search_results),
            dtype=np.float64,
            count=trajectory.stored_transition_count,
        )
        root_values = np.ascontiguousarray(root_values64, dtype=np.float32)
        predicted_values64 = np.asarray(
            trajectory.predicted_values, dtype=np.float64
        )
        predicted_values = np.ascontiguousarray(
            predicted_values64, dtype=np.float32
        )
        value_targets, value_valid_mask = self._build_value_target_table(
            rewards64,
            root_values64,
            terminated=trajectory.terminated,
        )
        priority_targets, priority_valid_mask = self._build_value_target_table(
            rewards64,
            predicted_values64,
            terminated=trajectory.terminated,
        )
        valid = priority_valid_mask[: len(trajectory)]
        initial_priority = self._priority_epsilon
        if np.any(valid):
            initial_errors = np.abs(
                predicted_values64[: len(trajectory)][valid]
                - priority_targets[: len(trajectory)][valid]
            )
            initial_priority += float(initial_errors.max())

        arrays = (
            frames,
            actions,
            rewards,
            policy_targets,
            root_values,
            predicted_values,
            value_targets,
            value_valid_mask,
        )
        for array in arrays:
            array.setflags(write=False)

        return _StoredTrajectory(
            environment_index=trajectory.environment_index,
            episode_id=trajectory.episode_id,
            block_id=trajectory.block_id,
            stack_size=trajectory.stack_size,
            sampleable_transition_count=len(trajectory),
            lookahead_steps=trajectory.lookahead_steps,
            terminated=trajectory.terminated,
            truncated=trajectory.truncated,
            frames=frames,
            actions=actions,
            rewards=rewards,
            policy_targets=policy_targets,
            root_values=root_values,
            predicted_values=predicted_values,
            initial_priority=initial_priority,
            value_targets=value_targets,
            value_valid_mask=value_valid_mask,
        )

    def _build_value_target_table(
        self,
        rewards: np.ndarray,
        root_values: np.ndarray,
        *,
        terminated: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute fixed-horizon values for every stored state."""
        stored_count = int(rewards.shape[0])
        bootstrap_count = max(0, stored_count - self.td_steps)
        values64 = np.zeros(stored_count + 1, dtype=np.float64)
        valid_mask = np.zeros(stored_count + 1, dtype=np.bool_)

        if terminated:
            # Terminal tail states have valid partial returns. Zero padding
            # represents the absent rewards after the episode boundary.
            padded_rewards = np.pad(rewards, (0, self.td_steps - 1))
            reward_windows = np.lib.stride_tricks.sliding_window_view(
                padded_rewards, self.td_steps
            )[:stored_count]
            values64[:stored_count] = reward_windows @ self._reward_discounts
            valid_mask[:] = True
        elif bootstrap_count:
            # Lookahead makes every valid nonterminal reward window complete.
            # Do not calculate targets for trailing context positions that
            # cannot bootstrap and will always be masked.
            reward_windows = np.lib.stride_tricks.sliding_window_view(
                rewards, self.td_steps
            )[:bootstrap_count]
            values64[:bootstrap_count] = reward_windows @ self._reward_discounts
            valid_mask[:bootstrap_count] = True

        if bootstrap_count:
            values64[:bootstrap_count] += (
                self._bootstrap_discount
                * root_values[self.td_steps : self.td_steps + bootstrap_count]
            )
        return values64.astype(np.float32), valid_mask

    def _validate_sample_request(self, batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._transition_count:
            raise ValueError(
                f"cannot sample {batch_size} unique transitions from "
                f"a replay buffer containing {self._transition_count}"
            )

    def _sample_context(
        self, batch_size: int
    ) -> tuple[
        list[tuple[_StoredTrajectory, int]],
        np.ndarray,
        np.ndarray,
    ]:
        probabilities = self._priorities.copy()
        probabilities /= probabilities.sum()
        flat_indices = self._rng.choice(
            self._transition_count,
            size=batch_size,
            replace=False,
            p=probabilities,
        )
        sampled_probabilities = probabilities[flat_indices]
        importance_weights = 1.0 / (self._transition_count * sampled_probabilities)
        importance_weights /= importance_weights.max()
        # EfficientZero V2 Atari floors normalized importance weights so
        # highly probable samples still contribute meaningfully to the loss.
        np.clip(importance_weights, 0.1, 1.0, out=importance_weights)
        return (
            self._locations_for_indices(flat_indices),
            self._transition_ids[flat_indices],
            importance_weights,
        )

    def _batch_array_specs(
        self,
        batch_size: int,
        *,
        include_value_bootstraps: bool,
    ) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        assert self._action_space_size is not None
        assert self._stack_size is not None
        assert self._frame_shape is not None
        states = self.unroll_steps + 1
        frame_spec = (
            (
                batch_size,
                self._stack_size + self.unroll_steps,
                *self._frame_shape,
            ),
            np.dtype(np.uint8),
        )
        specs = {
            "frames": frame_spec,
            "actions": (
                (batch_size, self.unroll_steps, 1),
                np.dtype(np.int64),
            ),
            "rewards": (
                (batch_size, self.unroll_steps),
                np.dtype(np.float32),
            ),
            "policy_targets": (
                (batch_size, states, self._action_space_size),
                np.dtype(np.float32),
            ),
            "value_targets": ((batch_size, states), np.dtype(np.float32)),
            "action_mask": (
                (batch_size, self.unroll_steps),
                np.dtype(np.bool_),
            ),
            "policy_mask": ((batch_size, states), np.dtype(np.bool_)),
            "value_mask": ((batch_size, states), np.dtype(np.bool_)),
            "reanalysis_frames": (
                (
                    batch_size,
                    self._stack_size + self.unroll_steps + self.td_steps,
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
            "transition_ages": ((batch_size,), np.dtype(np.int64)),
        }
        if include_value_bootstraps:
            del specs["frames"]
        else:
            for name in (
                "reanalysis_frames",
                "value_bootstrap_values",
                "value_bootstrap_discounts",
                "value_bootstrap_mask",
            ):
                del specs[name]
        return specs

    def _allocate_batch_arrays(
        self,
        batch_size: int,
        *,
        include_value_bootstraps: bool,
        pin_memory: bool = False,
    ) -> dict[str, np.ndarray]:
        specs = self._batch_array_specs(
            batch_size,
            include_value_bootstraps=include_value_bootstraps,
        )
        if not pin_memory:
            arrays = {
                name: np.empty(shape, dtype=dtype)
                for name, (shape, dtype) in specs.items()
            }
        else:
            arrays = {
                name: torch.empty(
                    shape,
                    dtype=torch.from_numpy(np.empty(0, dtype=dtype)).dtype,
                    pin_memory=True,
                ).numpy()
                for name, (shape, dtype) in specs.items()
            }
        shared_frames = arrays.get("reanalysis_frames")
        if shared_frames is not None:
            frame_count = self._stack_size + self.unroll_steps
            arrays["frames"] = shared_frames[:, :frame_count]
            arrays["value_bootstrap_frames"] = shared_frames[
                :, self.td_steps : self.td_steps + frame_count
            ]
        return arrays

    @staticmethod
    def _batch_from_arrays(arrays: Mapping[str, np.ndarray]) -> ReplayBatch:
        def optional_tensor(name: str) -> Tensor | None:
            array = arrays.get(name)
            return None if array is None else torch.from_numpy(array)

        shared_array = arrays.get("reanalysis_frames")
        if shared_array is None:
            shared_frames = None
            frames = torch.from_numpy(arrays["frames"])
            bootstrap_frames = optional_tensor("value_bootstrap_frames")
        else:
            shared_frames = torch.from_numpy(shared_array)
            frame_count = arrays["frames"].shape[1]
            bootstrap_count = arrays["value_bootstrap_frames"].shape[1]
            bootstrap_offset = shared_array.shape[1] - bootstrap_count
            frames = shared_frames[:, :frame_count]
            bootstrap_frames = shared_frames[
                :, bootstrap_offset : bootstrap_offset + bootstrap_count
            ]

        return ReplayBatch(
            frames=frames,
            actions=torch.from_numpy(arrays["actions"]),
            rewards=torch.from_numpy(arrays["rewards"]),
            policy_targets=torch.from_numpy(arrays["policy_targets"]),
            value_targets=torch.from_numpy(arrays["value_targets"]),
            action_mask=torch.from_numpy(arrays["action_mask"]),
            policy_mask=torch.from_numpy(arrays["policy_mask"]),
            value_mask=torch.from_numpy(arrays["value_mask"]),
            indices=torch.from_numpy(arrays["indices"]),
            importance_weights=torch.from_numpy(arrays["importance_weights"]),
            value_bootstrap_frames=bootstrap_frames,
            value_bootstrap_values=optional_tensor("value_bootstrap_values"),
            value_bootstrap_discounts=optional_tensor("value_bootstrap_discounts"),
            value_bootstrap_mask=optional_tensor("value_bootstrap_mask"),
            reanalysis_frames=shared_frames,
            transition_ages=torch.from_numpy(arrays["transition_ages"]),
        )

    def _locations_for_indices(
        self, flat_indices: np.ndarray
    ) -> list[tuple[_StoredTrajectory, int]]:
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
        locations: list[tuple[_StoredTrajectory, int]],
        *,
        transition_ids: np.ndarray,
        importance_weights: np.ndarray,
    ) -> None:
        """Fill a complete batch from insertion-time prepared arrays."""
        assert self._stack_size is not None

        stack_size = self._stack_size
        unroll_steps = self.unroll_steps
        state_count = unroll_steps + 1
        full_frame_count = stack_size + unroll_steps
        frames = arrays["frames"]
        actions = arrays["actions"]
        rewards = arrays["rewards"]
        policy_targets = arrays["policy_targets"]
        value_targets = arrays["value_targets"]
        action_mask = arrays["action_mask"]
        policy_mask = arrays["policy_mask"]
        value_mask = arrays["value_mask"]
        reanalysis_frames = arrays.get("reanalysis_frames")
        value_bootstrap_frames = arrays.get("value_bootstrap_frames")
        value_bootstrap_values = arrays.get("value_bootstrap_values")
        value_bootstrap_discounts = arrays.get("value_bootstrap_discounts")
        value_bootstrap_mask = arrays.get("value_bootstrap_mask")

        for batch_index, (trajectory, start) in enumerate(locations):
            stored_count = trajectory.stored_transition_count
            action_count = min(unroll_steps, stored_count - start)
            frame_count = stack_size + action_count
            if reanalysis_frames is None:
                frames[batch_index, :frame_count] = trajectory.frames[
                    start : start + frame_count
                ]
                if frame_count < full_frame_count:
                    frames[batch_index, frame_count:] = frames[
                        batch_index, frame_count - 1
                    ]
            else:
                available_frames = min(
                    reanalysis_frames.shape[1],
                    trajectory.frames.shape[0] - start,
                )
                reanalysis_frames[batch_index, :available_frames] = trajectory.frames[
                    start : start + available_frames
                ]
                if available_frames < reanalysis_frames.shape[1]:
                    reanalysis_frames[batch_index, available_frames:] = (
                        reanalysis_frames[batch_index, available_frames - 1]
                    )

            if action_count < unroll_steps:
                actions[batch_index].fill(0)
                rewards[batch_index].fill(0)
                action_mask[batch_index].fill(False)
            else:
                action_mask[batch_index].fill(True)
            action_end = start + action_count
            actions[batch_index, :action_count, 0] = trajectory.actions[
                start:action_end
            ]
            rewards[batch_index, :action_count] = trajectory.rewards[start:action_end]
            if action_count < unroll_steps:
                action_mask[batch_index, :action_count] = True

            policy_count = min(state_count, stored_count - start)
            if policy_count < state_count:
                policy_targets[batch_index].fill(0)
                policy_mask[batch_index].fill(False)
            else:
                policy_mask[batch_index].fill(True)
            policy_targets[batch_index, :policy_count] = trajectory.policy_targets[
                start : start + policy_count
            ]
            if policy_count < state_count:
                policy_mask[batch_index, :policy_count] = True

            value_count = min(state_count, stored_count + 1 - start)
            if value_count < state_count:
                value_targets[batch_index].fill(0)
                value_mask[batch_index].fill(False)
            value_targets[batch_index, :value_count] = trajectory.value_targets[
                start : start + value_count
            ]
            value_mask[batch_index, :value_count] = trajectory.value_valid_mask[
                start : start + value_count
            ]

            if value_bootstrap_frames is not None:
                assert value_bootstrap_values is not None
                assert value_bootstrap_discounts is not None
                assert value_bootstrap_mask is not None
                bootstrap_start = start + self.td_steps
                bootstrap_count = min(
                    state_count, max(0, stored_count - bootstrap_start)
                )
                bootstrap_frame_count = (
                    bootstrap_count + stack_size - 1 if bootstrap_count else 0
                )
                if (
                    reanalysis_frames is None
                    and bootstrap_frame_count < full_frame_count
                ):
                    value_bootstrap_frames[batch_index].fill(0)
                if bootstrap_count < state_count:
                    value_bootstrap_values[batch_index].fill(0)
                    value_bootstrap_discounts[batch_index].fill(0)
                    value_bootstrap_mask[batch_index].fill(False)
                else:
                    value_bootstrap_discounts[batch_index].fill(
                        self._bootstrap_discount
                    )
                    value_bootstrap_mask[batch_index].fill(True)
                if bootstrap_count:
                    if reanalysis_frames is None:
                        value_bootstrap_frames[batch_index, :bootstrap_frame_count] = (
                            trajectory.frames[
                                bootstrap_start : bootstrap_start
                                + bootstrap_frame_count
                            ]
                        )
                    value_bootstrap_values[batch_index, :bootstrap_count] = (
                        trajectory.root_values[
                            bootstrap_start : bootstrap_start + bootstrap_count
                        ]
                    )
                    if bootstrap_count < state_count:
                        value_bootstrap_discounts[batch_index, :bootstrap_count] = (
                            self._bootstrap_discount
                        )
                        value_bootstrap_mask[batch_index, :bootstrap_count] = True

        arrays["indices"][:] = transition_ids
        arrays["importance_weights"][:] = importance_weights
        # Age is the number of newer transitions, so the newest transition
        # has age zero and exactly ``freshness_threshold`` transitions satisfy
        # ``age < freshness_threshold``.
        arrays["transition_ages"][:] = self._next_transition_id - 1 - transition_ids

    @staticmethod
    def _trajectory_key(
        trajectory: GameTrajectory | _StoredTrajectory,
    ) -> tuple[int, int, int]:
        return (
            trajectory.environment_index,
            trajectory.episode_id,
            trajectory.block_id,
        )


__all__ = ["FIFOReplayBuffer", "ReplayAddResult", "ReplayBatch"]
