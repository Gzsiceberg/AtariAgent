"""FIFO trajectory replay buffer with prioritized tensor-batch sampling."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import warnings

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
    initial_priorities: np.ndarray
    value_targets: np.ndarray
    value_valid_mask: np.ndarray

    def __len__(self) -> int:
        return self.sampleable_transition_count

    @property
    def stored_transition_count(self) -> int:
        return int(self.actions.shape[0])


class FIFOReplayBuffer:
    """Store prepared trajectories with FIFO prioritized sampling.

    Target horizons and discount are fixed for the buffer lifetime.
    By default both modes use V2 maximum-priority insertion: every new
    trajectory start receives max(current buffer maximum, trajectory error
    maximum), with a buffer maximum of 1 only when empty. Stored
    initial_priorities retain the individual prediction/bootstrap errors.
    use_max_priority=True instead uses only the live buffer maximum (V1),
    independently of the sampling mode.
    V1 uses configurable alpha/beta and unfloored importance weights;
    V2 forces alpha=beta=1 and a 0.1 normalized importance-weight floor.
    Trailing lookahead transitions remain local target context and are not
    replay starts. Reanalysis remains outside this buffer.
    """

    def __init__(
        self,
        max_transitions: int,
        *,
        unroll_steps: int = 5,
        td_steps: int = 5,
        discount: float = 0.997,
        priority_weight_clip: float = 0.0,
        use_max_priority: bool = False,
        treat_truncations_as_terminal: bool = False,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
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
        if not isinstance(treat_truncations_as_terminal, bool):
            raise TypeError("treat_truncations_as_terminal must be a boolean")
        if not isinstance(use_max_priority, bool):
            raise TypeError("use_max_priority must be a boolean")
        for value, name in (
            (priority_alpha, "priority_alpha"),
            (priority_beta, "priority_beta"),
            (priority_weight_clip, "priority_weight_clip"),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not np.isfinite(priority_epsilon) or priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be finite and positive")

        self.max_transitions = max_transitions
        self._unroll_steps = unroll_steps
        self._td_steps = td_steps
        self._discount = float(discount)
        self._priority_weight_clip = float(priority_weight_clip)
        self._use_max_priority = use_max_priority
        self._treat_truncations_as_terminal = treat_truncations_as_terminal
        self._priority_alpha = float(priority_alpha)
        self._priority_beta = float(priority_beta)
        self._priority_epsilon = float(priority_epsilon)
        self._reward_discounts = self.discount ** np.arange(
            self.td_steps, dtype=np.float64
        )
        self._bootstrap_discount = float(self.discount**self.td_steps)
        self._trajectories: deque[_StoredTrajectory] = deque()
        self._reanalysis_state_ids: deque[np.ndarray] = deque()
        self._reanalysis_state_registry: dict[
            tuple[int, int, int, int], tuple[int, int]
        ] = {}
        self._next_reanalysis_state_id = 0
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

    def state_dict(self, *, tensor_arrays: bool = False) -> dict[str, object]:
        """Return replay state, optionally with zero-copy tensors for torch.save.

        NumPy arrays are pickled into an in-memory buffer by torch.save, causing
        large RAM spikes for frame data. Tensor storage is streamed separately.
        Tensor views may alias read-only replay arrays: serialize them only while
        replay is quiescent, and never mutate the returned tensors.
        """
        def encode(value: object) -> object:
            if not tensor_arrays or not isinstance(value, np.ndarray):
                return value
            # PyTorch cannot express NumPy's read-only flag. Saving only reads
            # this shared storage; copying it would defeat the memory benefit.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The given NumPy array is not writable",
                    category=UserWarning,
                )
                return torch.from_numpy(value)

        trajectory_fields = tuple(_StoredTrajectory.__dataclass_fields__)
        return {
            "version": 1,
            "treat_truncations_as_terminal": self._treat_truncations_as_terminal,
            "max_transitions": self.max_transitions,
            "unroll_steps": self.unroll_steps,
            "td_steps": self.td_steps,
            "discount": self.discount,
            "priority_epsilon": self._priority_epsilon,
            "trajectories": [
                {name: encode(getattr(trajectory, name)) for name in trajectory_fields}
                for trajectory in self._trajectories
            ],
            "transition_ids": encode(self._transition_ids),
            "priorities": encode(self._priorities),
            "next_transition_id": self._next_transition_id,
            "transition_count": self._transition_count,
            "action_space_size": self._action_space_size,
            "stack_size": self._stack_size,
            "frame_shape": self._frame_shape,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore replay data produced by :meth:`state_dict`."""
        if not isinstance(state, Mapping):
            raise TypeError("replay state must be a mapping")
        if state.get("version") != 1:
            raise ValueError("unsupported replay state version")
        expected_configuration = {
            "max_transitions": self.max_transitions,
            "unroll_steps": self.unroll_steps,
            "td_steps": self.td_steps,
            "discount": self.discount,
            "priority_epsilon": self._priority_epsilon,
        }
        for name, expected in expected_configuration.items():
            if state.get(name) != expected:
                raise ValueError(
                    f"replay state {name} does not match the configured buffer"
                )

        # Older snapshots used masked truncation tails. Reject mismatched modes
        # because targets and initial priorities are precomputed in the snapshot.
        if state.get("treat_truncations_as_terminal", False) != self._treat_truncations_as_terminal:
            raise ValueError(
                "replay state treat_truncations_as_terminal does not match the configured buffer"
            )

        raw_trajectories = state.get("trajectories")
        if not isinstance(raw_trajectories, list):
            raise TypeError("replay trajectories must be a list")
        trajectory_fields = tuple(_StoredTrajectory.__dataclass_fields__)
        array_fields = {
            "frames",
            "actions",
            "rewards",
            "policy_targets",
            "root_values",
            "predicted_values",
            "initial_priorities",
            "value_targets",
            "value_valid_mask",
        }
        trajectories: deque[_StoredTrajectory] = deque()
        for raw in raw_trajectories:
            if not isinstance(raw, Mapping):
                raise TypeError("each replay trajectory must be a mapping")
            values = dict(raw)
            if set(values) != set(trajectory_fields):
                raise ValueError("replay trajectory fields are invalid")
            for name in array_fields:
                array = np.ascontiguousarray(np.asarray(values[name]))
                array.setflags(write=False)
                values[name] = array
            trajectories.append(_StoredTrajectory(**values))

        transition_ids = np.ascontiguousarray(
            np.asarray(state.get("transition_ids"), dtype=np.int64)
        )
        priorities = np.ascontiguousarray(
            np.asarray(state.get("priorities"), dtype=np.float64)
        )
        transition_count = state.get("transition_count")
        next_transition_id = state.get("next_transition_id")
        if isinstance(transition_count, bool) or not isinstance(transition_count, int):
            raise TypeError("replay transition_count must be an integer")
        if isinstance(next_transition_id, bool) or not isinstance(
            next_transition_id, int
        ):
            raise TypeError("replay next_transition_id must be an integer")
        if transition_count != sum(map(len, trajectories)):
            raise ValueError("replay transition count does not match trajectories")
        if transition_count > self.max_transitions:
            raise ValueError("replay state exceeds the configured capacity")
        if transition_ids.shape != (transition_count,):
            raise ValueError("replay transition IDs have an invalid shape")
        if priorities.shape != (transition_count,):
            raise ValueError("replay priorities have an invalid shape")
        if priorities.size and (
            not np.all(np.isfinite(priorities)) or np.any(priorities <= 0.0)
        ):
            raise ValueError("replay priorities must be finite and positive")
        if transition_ids.size > 1 and np.any(np.diff(transition_ids) != 1):
            raise ValueError("replay transition IDs must be contiguous")
        if transition_ids.size and next_transition_id <= int(transition_ids[-1]):
            raise ValueError("replay next transition ID is invalid")

        trajectory_keys = {self._trajectory_key(item) for item in trajectories}
        if len(trajectory_keys) != len(trajectories):
            raise ValueError("replay trajectory identities must be unique")
        reanalysis_state_ids: deque[np.ndarray] = deque()
        reanalysis_state_registry: dict[
            tuple[int, int, int, int], tuple[int, int]
        ] = {}
        next_reanalysis_state_id = 0
        for trajectory in trajectories:
            state_ids, next_reanalysis_state_id = self._intern_state_ids(
                trajectory,
                reanalysis_state_registry,
                next_reanalysis_state_id,
            )
            reanalysis_state_ids.append(state_ids)
        rng = np.random.default_rng()
        try:
            rng.bit_generator.state = state["rng_state"]  # type: ignore[assignment]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("replay RNG state is invalid") from error

        self._trajectories = trajectories
        self._reanalysis_state_ids = reanalysis_state_ids
        self._reanalysis_state_registry = reanalysis_state_registry
        self._next_reanalysis_state_id = next_reanalysis_state_id
        self._transition_ids = transition_ids
        self._priorities = priorities
        self._next_transition_id = next_transition_id
        self._trajectory_keys = trajectory_keys
        self._transition_count = transition_count
        self._action_space_size = state.get("action_space_size")  # type: ignore[assignment]
        self._stack_size = state.get("stack_size")  # type: ignore[assignment]
        raw_frame_shape = state.get("frame_shape")
        self._frame_shape = (
            None if raw_frame_shape is None else tuple(raw_frame_shape)  # type: ignore[arg-type]
        )
        self._rng = rng

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

        # Match EZ V2 save_trajectory: use the live buffer maximum (not a
        # historical high-water mark) and the maximum error of replay starts,
        # excluding lookahead. Errors already include epsilon; do not add it
        # again. Compute before eviction so insertion sees the current buffer.
        buffer_maximum = (
            float(self._priorities.max()) if self._transition_count else 1.0
        )
        insertion_priority = (
            buffer_maximum
            if self._use_max_priority
            else max(buffer_maximum, float(stored.initial_priorities.max()))
        )
        insertion_priorities = np.full(
            trajectory_length, insertion_priority, dtype=np.float64
        )

        evicted_trajectories = 0
        evicted_transitions = 0
        state_ids, self._next_reanalysis_state_id = self._intern_state_ids(
            stored,
            self._reanalysis_state_registry,
            self._next_reanalysis_state_id,
        )
        while self._transition_count + trajectory_length > self.max_transitions:
            evicted = self._trajectories.popleft()
            evicted_state_ids = self._reanalysis_state_ids.popleft()
            self._release_state_ids(evicted, evicted_state_ids)
            self._trajectory_keys.remove(self._trajectory_key(evicted))
            self._transition_count -= len(evicted)
            evicted_trajectories += 1
            evicted_transitions += len(evicted)

        if evicted_transitions:
            self._transition_ids = self._transition_ids[evicted_transitions:]
            self._priorities = self._priorities[evicted_transitions:]

        # Existing priorities remain unchanged; only new starts receive the
        # shared maximum. Learner updates subsequently assign individual errors.
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
                insertion_priorities,
            )
        )

        if self._action_space_size is None:
            self._action_space_size = action_space_size
            self._stack_size = stored.stack_size
            self._frame_shape = frame_shape
        self._trajectories.append(stored)
        self._reanalysis_state_ids.append(state_ids)
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
        priority_beta: float | None = None,
    ) -> ReplayBatch:
        """Prioritize unique starts and copy their prepared local context."""
        self._validate_sample_request(batch_size)
        if not isinstance(include_value_bootstraps, bool):
            raise TypeError("include_value_bootstraps must be a boolean")
        if not isinstance(pin_memory, bool):
            raise TypeError("pin_memory must be a boolean")
        requested_beta = self._priority_beta if priority_beta is None else priority_beta
        if not np.isfinite(requested_beta) or not 0.0 <= requested_beta <= 1.0:
            raise ValueError("priority_beta must be finite and in [0, 1]")
        resolved_beta = float(requested_beta)
        assert self._action_space_size is not None

        locations, transition_ids, importance_weights = self._sample_context(
            batch_size, priority_beta=resolved_beta
        )
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
            len(result.target_policy) for result in trajectory.search_results
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
        policy_targets = np.ascontiguousarray(
            np.asarray(
                [
                    result.target_policy
                    for result in trajectory.search_results
                ],
                dtype=np.float32,
            )
        )
        if policy_targets.shape != (
            trajectory.stored_transition_count,
            action_space_size,
        ):
            raise ValueError("policy targets have an invalid shape")
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
        # Preserve the environment flags; only target construction treats a
        # timeout as terminal when V1-style zero continuation is requested.
        terminal_target = trajectory.terminated or (
            self._treat_truncations_as_terminal and trajectory.truncated
        )
        value_targets, value_valid_mask = self._build_value_target_table(
            rewards64,
            root_values64,
            terminated=terminal_target,
        )
        # Initial errors bootstrap from network predictions, not MCTS values.
        priority_targets, priority_valid_mask = self._build_value_target_table(
            rewards64,
            predicted_values64,
            terminated=terminal_target,
        )
        valid = priority_valid_mask[: len(trajectory)]
        initial_priorities = np.full(
            len(trajectory), self._priority_epsilon, dtype=np.float64
        )
        initial_priorities[valid] += np.abs(
            predicted_values64[: len(trajectory)][valid]
            - priority_targets[: len(trajectory)][valid]
        )

        arrays = (
            frames,
            actions,
            rewards,
            policy_targets,
            root_values,
            predicted_values,
            initial_priorities,
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
            initial_priorities=initial_priorities,
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
        self,
        batch_size: int,
        *,
        priority_beta: float | None = None,
    ) -> tuple[
        list[tuple[_StoredTrajectory, int]],
        np.ndarray,
        np.ndarray,
    ]:
        resolved_beta = self._priority_beta if priority_beta is None else priority_beta
        probabilities = self._priorities**self._priority_alpha
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
        ) ** -resolved_beta
        importance_weights /= importance_weights.max()
        if self._priority_weight_clip > 0.0:
            np.clip(
                importance_weights, self._priority_weight_clip, 1.0,
                out=importance_weights,
            )
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
            "value_bootstrap_state_ids": (
                (batch_size, states),
                np.dtype(np.int64),
            ),
            "indices": ((batch_size,), np.dtype(np.int64)),
            "importance_weights": ((batch_size,), np.dtype(np.float32)),
            "transition_ages": ((batch_size,), np.dtype(np.int64)),
            "reanalysis_state_ids": (
                (batch_size, states),
                np.dtype(np.int64),
            ),
        }
        if include_value_bootstraps:
            del specs["frames"]
        else:
            for name in (
                "reanalysis_frames",
                "value_bootstrap_values",
                "value_bootstrap_discounts",
                "value_bootstrap_mask",
                "value_bootstrap_state_ids",
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
            value_bootstrap_state_ids=optional_tensor(
                "value_bootstrap_state_ids"
            ),
            reanalysis_frames=shared_frames,
            transition_ages=torch.from_numpy(arrays["transition_ages"]),
            reanalysis_state_ids=torch.from_numpy(
                arrays["reanalysis_state_ids"]
            ),
        )

    def _locations_for_indices(
        self, flat_indices: np.ndarray
    ) -> list[tuple[_StoredTrajectory, np.ndarray, int]]:
        """Resolve flat replay offsets with vectorized cumulative boundaries."""
        trajectories = tuple(self._trajectories)
        state_ids = tuple(self._reanalysis_state_ids)
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
            (
                trajectories[int(trajectory_index)],
                state_ids[int(trajectory_index)],
                int(position),
            )
            for trajectory_index, position in zip(
                trajectory_indices, positions, strict=True
            )
        ]

    def _fill_batch_arrays(
        self,
        arrays: Mapping[str, np.ndarray],
        locations: list[tuple[_StoredTrajectory, np.ndarray, int]],
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
        value_bootstrap_state_ids = arrays.get("value_bootstrap_state_ids")
        reanalysis_state_ids = arrays["reanalysis_state_ids"]

        for batch_index, (trajectory, trajectory_state_ids, start) in enumerate(
            locations
        ):
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
            reanalysis_state_ids[batch_index].fill(-1)
            reanalysis_state_ids[batch_index, :policy_count] = (
                trajectory_state_ids[start : start + policy_count]
            )

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
                assert value_bootstrap_state_ids is not None
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
                value_bootstrap_state_ids[batch_index].fill(-1)
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
                    value_bootstrap_state_ids[
                        batch_index, :bootstrap_count
                    ] = trajectory_state_ids[
                        bootstrap_start : bootstrap_start + bootstrap_count
                    ]
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
    def _logical_state_key(
        trajectory: _StoredTrajectory,
        position: int,
    ) -> tuple[int, int, int, int]:
        block_offset, block_position = divmod(position, len(trajectory))
        return (
            trajectory.environment_index,
            trajectory.episode_id,
            trajectory.block_id + block_offset,
            block_position,
        )

    @classmethod
    def _intern_state_ids(
        cls,
        trajectory: _StoredTrajectory,
        registry: dict[tuple[int, int, int, int], tuple[int, int]],
        next_state_id: int,
    ) -> tuple[np.ndarray, int]:
        """Assign compact IDs while sharing canonical overlapping states."""
        state_ids = np.empty(trajectory.stored_transition_count, dtype=np.int64)
        for position in range(trajectory.stored_transition_count):
            key = cls._logical_state_key(trajectory, position)
            registered = registry.get(key)
            if registered is None:
                state_id = next_state_id
                next_state_id += 1
                registry[key] = (state_id, 1)
            else:
                state_id, references = registered
                registry[key] = (state_id, references + 1)
            state_ids[position] = state_id
        state_ids.setflags(write=False)
        return state_ids, next_state_id

    def _release_state_ids(
        self,
        trajectory: _StoredTrajectory,
        state_ids: np.ndarray,
    ) -> None:
        """Release canonical identities once no stored block references them."""
        for position, raw_state_id in enumerate(state_ids):
            key = self._logical_state_key(trajectory, position)
            state_id, references = self._reanalysis_state_registry[key]
            if state_id != int(raw_state_id):
                raise RuntimeError("reanalysis state identity is inconsistent")
            if references == 1:
                del self._reanalysis_state_registry[key]
            else:
                self._reanalysis_state_registry[key] = (
                    state_id,
                    references - 1,
                )

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
