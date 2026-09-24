"""Append-only episode replay with prioritized tensor-batch sampling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import warnings

import numpy as np
import torch
from torch import Tensor

from .replay_batch import ReplayBatch
from .selfplay import GameTrajectory


@dataclass(frozen=True, slots=True)
class ReplayAddResult:
    """Insertion statistics; legacy eviction counters are always zero."""

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
    last_block_id: int
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
    transition_ids: np.ndarray
    value_targets: np.ndarray
    value_valid_mask: np.ndarray

    def __len__(self) -> int:
        return self.sampleable_transition_count

    @property
    def stored_transition_count(self) -> int:
        return int(self.actions.shape[0])


class FIFOReplayBuffer:
    """Store one contiguous NumPy trajectory per episode segment.

    The historical class name and ``max_transitions`` argument remain for
    compatibility. The latter is a reporting budget, not a capacity: replay
    grows without eviction. Life-loss terminals separate episode segments.

    Target horizons and discount are fixed for the buffer lifetime.
    By default both modes use V2 maximum-priority insertion: every new
    trajectory start receives max(current buffer maximum, trajectory error
    maximum), with a buffer maximum of 1 only when empty. Stored
    initial_priorities retain the individual prediction/bootstrap errors.
    use_max_priority=True instead uses only the live buffer maximum (V1),
    independently of the sampling mode.
    V1 uses configurable alpha/beta and unfloored importance weights;
    V2 forces alpha=beta=1 and a 0.1 normalized importance-weight floor.
    ``unroll_steps`` also sets the TD-bootstrap horizon. Trailing lookahead
    transitions are inactive until their owning continuation block arrives.
    Continuations replace the episode arrays on insertion, with a temporary
    terminal at the new sampleable end. Direct NumPy slot-to-episode/offset
    tables make sampling independent of insertion-block count. No original
    chunk payloads or secondary merged copies are retained.
    """

    def __init__(
        self,
        max_transitions: int,
        *,
        unroll_steps: int = 5,
        discount: float = 0.997,
        priority_weight_clip: float = 0.0,
        use_max_priority: bool = False,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        priority_epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if isinstance(max_transitions, bool) or not isinstance(max_transitions, int):
            raise TypeError("max_transitions must be an integer")
        if max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        if isinstance(unroll_steps, bool) or not isinstance(unroll_steps, int):
            raise TypeError("unroll_steps must be an integer")
        if unroll_steps <= 0:
            raise ValueError("unroll_steps must be positive")
        if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")
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
        self._discount = float(discount)
        self._priority_weight_clip = float(priority_weight_clip)
        self._use_max_priority = use_max_priority
        self._priority_alpha = float(priority_alpha)
        self._priority_beta = float(priority_beta)
        self._priority_epsilon = float(priority_epsilon)
        self._reward_discounts = self.discount ** np.arange(
            self.unroll_steps, dtype=np.float64
        )
        self._bootstrap_discount = float(self.discount**self.unroll_steps)
        self._trajectories: list[_StoredTrajectory] = []
        self._latest_trajectory_rows: dict[int, int] = {}
        self._trajectory_rows = np.empty(0, dtype=np.int64)
        self._trajectory_offsets = np.empty(0, dtype=np.int64)
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
        """Number of stored episode segments, not incoming blocks."""
        return len(self._trajectories)

    @property
    def unroll_steps(self) -> int:
        return self._unroll_steps

    @property
    def discount(self) -> float:
        return self._discount

    @property
    def action_space_size(self) -> int | None:
        return self._action_space_size

    @property
    def utilization(self) -> float:
        """Fraction of the reporting budget used; may exceed one."""
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
            "version": 4,
            "max_transitions": self.max_transitions,
            "unroll_steps": self.unroll_steps,
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
        # Older snapshots used different boundary/target semantics.
        # Reject rather than silently train on stale targets or priorities.
        if state.get("version") != 4:
            raise ValueError("unsupported replay state version")
        expected_configuration = {
            "max_transitions": self.max_transitions,
            "unroll_steps": self.unroll_steps,
            "discount": self.discount,
            "priority_epsilon": self._priority_epsilon,
        }
        for name, expected in expected_configuration.items():
            if state.get(name) != expected:
                raise ValueError(
                    f"replay state {name} does not match the configured buffer"
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
            "transition_ids",
            "value_targets",
            "value_valid_mask",
        }
        trajectories: list[_StoredTrajectory] = []
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
        if transition_ids.shape != (transition_count,):
            raise ValueError("replay transition IDs have an invalid shape")
        if priorities.shape != (transition_count,):
            raise ValueError("replay priorities have an invalid shape")
        if priorities.size and (
            not np.all(np.isfinite(priorities)) or np.any(priorities <= 0.0)
        ):
            raise ValueError("replay priorities must be finite and positive")
        if not np.array_equal(transition_ids, np.arange(transition_count)):
            raise ValueError("replay transition IDs must be contiguous from zero")
        if next_transition_id != transition_count:
            raise ValueError("replay next transition ID is invalid")

        # Build the hot-path tables once, not at sampling time. Episode IDs
        # can interleave globally, so lengths/cumulative offsets are insufficient.
        rows = np.empty(transition_count, dtype=np.int64)
        offsets = np.empty(transition_count, dtype=np.int64)
        seen = np.zeros(transition_count, dtype=np.bool_)
        trajectory_keys: set[tuple[int, int, int]] = set()
        latest_rows: dict[int, int] = {}
        for row, trajectory in enumerate(trajectories):
            ids = trajectory.transition_ids
            if (ids.dtype != np.int64 or ids.shape != (len(trajectory),)
                    or np.any(ids < 0) or np.any(ids >= transition_count)
                    or np.unique(ids).size != ids.size or np.any(seen[ids])):
                raise ValueError("episode transition IDs are invalid or duplicated")
            seen[ids] = True
            rows[ids] = row
            offsets[ids] = np.arange(len(trajectory))
            if trajectory.last_block_id < trajectory.block_id:
                raise ValueError("episode block range is invalid")
            for block_id in range(trajectory.block_id, trajectory.last_block_id + 1):
                key = (trajectory.environment_index, trajectory.episode_id, block_id)
                if key in trajectory_keys:
                    raise ValueError("replay trajectory identities must be unique")
                trajectory_keys.add(key)
            latest_rows[trajectory.environment_index] = row
        if not seen.all():
            raise ValueError("episode transition IDs do not cover replay")
        rng = np.random.default_rng()
        try:
            rng.bit_generator.state = state["rng_state"]  # type: ignore[assignment]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("replay RNG state is invalid") from error

        self._trajectories = trajectories
        self._latest_trajectory_rows = latest_rows
        self._trajectory_rows = rows
        self._trajectory_offsets = offsets
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
        """Append a block to its episode; concatenate only on this cold path."""
        trajectory_length = len(trajectory)

        key = self._trajectory_key(trajectory)
        if key in self._trajectory_keys:
            raise ValueError("trajectory identity already exists in replay")

        # Preparation and all validation happen before replay state is mutated.
        # Partial lookahead is stored but inactive until its owning block arrives.
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
        # again. Existing starts retain their individually updated priorities.
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

        row = self._latest_trajectory_rows.get(stored.environment_index)
        previous = self._trajectories[row] if row is not None else None
        joins = previous is not None and self._can_join(previous, stored)
        offset = len(previous) if joins else 0
        if joins:
            assert previous is not None
            prepared = self._concat_episode(previous, stored)
        else:
            row = len(self._trajectories)
            prepared = stored
        assert row is not None

        # All allocations/target construction precede mutation. Global slots
        # stay in insertion order even when episode continuations interleave.
        transition_ids = np.concatenate((self._transition_ids, stored.transition_ids))
        priorities = np.concatenate((self._priorities, insertion_priorities))
        rows = np.concatenate((
            self._trajectory_rows, np.full(trajectory_length, row, dtype=np.int64),
        ))
        offsets = np.concatenate((
            self._trajectory_offsets, np.arange(offset, offset + trajectory_length),
        ))
        self._transition_ids = transition_ids
        self._priorities = priorities
        self._trajectory_rows = rows
        self._trajectory_offsets = offsets
        self._next_transition_id += trajectory_length

        if self._action_space_size is None:
            self._action_space_size = action_space_size
            self._stack_size = stored.stack_size
            self._frame_shape = frame_shape
        if joins:
            self._trajectories[row] = prepared
        else:
            self._trajectories.append(prepared)
        self._latest_trajectory_rows[stored.environment_index] = row
        self._trajectory_keys.add(key)
        self._transition_count += trajectory_length
        return ReplayAddResult(
            added_trajectories=int(not joins),
            added_transitions=trajectory_length,
            evicted_trajectories=0,
            evicted_transitions=0,
        )

    @staticmethod
    def _can_join(previous: _StoredTrajectory, incoming: _StoredTrajectory) -> bool:
        return (
            previous.environment_index == incoming.environment_index
            and previous.episode_id == incoming.episode_id
            and previous.last_block_id + 1 == incoming.block_id
            # A terminal in older lookahead belongs to the arriving owner.
            and not ((previous.terminated or previous.truncated)
                     and previous.lookahead_steps == 0)
            and np.array_equal(
                previous.frames[len(previous) + previous.stack_size - 1],
                incoming.frames[incoming.stack_size - 1],
            )
        )

    def _concat_episode(
        self, previous: _StoredTrajectory, incoming: _StoredTrajectory,
    ) -> _StoredTrajectory:
        """Replace inactive lookahead with its owner and extend one episode.

        Only the new arrays survive insertion; there is no parallel chunk or
        merged-context store. Immutable old arrays remain safe for snapshots.
        """
        count = len(previous) + len(incoming)
        arrays = {
            name: np.concatenate((
                getattr(previous, name)[:len(previous)], getattr(incoming, name),
            ))
            for name in (
                "actions", "rewards", "policy_targets", "root_values",
                "predicted_values", "initial_priorities", "transition_ids",
            )
        }
        arrays["frames"] = np.concatenate((
            previous.frames[:len(previous) + previous.stack_size],
            incoming.frames[incoming.stack_size:],
        ))
        arrays["value_targets"], arrays["value_valid_mask"] = self._build_value_target_table(
            arrays["rewards"], arrays["root_values"], sampleable_count=count,
        )
        for array in arrays.values():
            array.setflags(write=False)
        return replace(
            previous, last_block_id=incoming.last_block_id,
            sampleable_transition_count=count, lookahead_steps=incoming.lookahead_steps,
            terminated=incoming.terminated, truncated=incoming.truncated, **arrays,
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
        """Prioritize unique starts and gather precomputed episode targets."""
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
        value_targets, value_valid_mask = self._build_value_target_table(
            rewards64, root_values64, sampleable_count=len(trajectory)
        )
        # Initial priorities also respect the temporary terminal.
        priority_targets, priority_valid_mask = self._build_value_target_table(
            rewards64, predicted_values64,
            sampleable_count=len(trajectory),
        )
        valid = priority_valid_mask[: len(trajectory)]
        initial_priorities = np.full(
            len(trajectory), self._priority_epsilon, dtype=np.float64
        )
        initial_priorities[valid] += np.abs(
            predicted_values64[: len(trajectory)][valid]
            - priority_targets[: len(trajectory)][valid]
        )

        transition_ids = np.arange(
            self._next_transition_id, self._next_transition_id + len(trajectory),
            dtype=np.int64,
        )
        arrays = (
            transition_ids,
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
            last_block_id=trajectory.block_id,
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
            transition_ids=transition_ids,
            value_targets=value_targets,
            value_valid_mask=value_valid_mask,
        )

    def _build_value_target_table(
        self,
        rewards: np.ndarray,
        root_values: np.ndarray,
        *,
        sampleable_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """TD returns inside the currently available merged trajectory.

        Its sampleable end is a zero-value terminal, real or temporary.
        Stored lookahead rewards remain inactive until their owner arrives.
        """
        stored_count = int(rewards.shape[0])
        bootstrap_count = max(0, sampleable_count - self.unroll_steps)
        values64 = np.zeros(stored_count + 1, dtype=np.float64)
        valid_mask = np.zeros(stored_count + 1, dtype=np.bool_)

        padded_rewards = np.pad(
            rewards[:sampleable_count], (0, self.unroll_steps - 1)
        )
        reward_windows = np.lib.stride_tricks.sliding_window_view(
            padded_rewards, self.unroll_steps
        )[:sampleable_count]
        values64[:sampleable_count] = reward_windows @ self._reward_discounts
        valid_mask[:sampleable_count] = True

        if bootstrap_count:
            values64[:bootstrap_count] += (
                self._bootstrap_discount
                * root_values[self.unroll_steps : self.unroll_steps + bootstrap_count]
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
        list[tuple[_StoredTrajectory, np.ndarray, int]],
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
            "reachable_mask": ((batch_size, states), np.dtype(np.bool_)),
            "terminal_mask": ((batch_size, states), np.dtype(np.bool_)),
            "reanalysis_frames": (
                (
                    batch_size,
                    self._stack_size + 2 * self.unroll_steps,
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
                :, self.unroll_steps : self.unroll_steps + frame_count
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
            reachable_mask=torch.from_numpy(arrays["reachable_mask"]),
            terminal_mask=torch.from_numpy(arrays["terminal_mask"]),
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
        """O(batch size) direct lookup; no scan, prefix sum, or key lookup."""
        rows = self._trajectory_rows[flat_indices].tolist()
        offsets = self._trajectory_offsets[flat_indices].tolist()
        return [
            (trajectory, trajectory.transition_ids, offset)
            for row, offset in zip(rows, offsets, strict=True)
            for trajectory in (self._trajectories[row],)
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
        reachable_mask = arrays["reachable_mask"]
        reachable_mask[:, 0] = True
        terminal_mask = arrays["terminal_mask"]
        terminal_mask.fill(False)
        action_mask = reachable_mask[:, 1:]
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
            block_count = len(trajectory)
            endpoint = block_count - start
            if endpoint <= unroll_steps:
                terminal_mask[batch_index, endpoint] = True
            # The first inactive lookahead state is a temporary terminal.
            # Merging a continuation moves this endpoint forward.
            action_count = min(unroll_steps, endpoint)
            frame_count = min(full_frame_count, block_count + stack_size - start)
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
                    block_count + stack_size - start,
                )
                reanalysis_frames[batch_index, :available_frames] = trajectory.frames[
                    start : start + available_frames
                ]
                if available_frames < reanalysis_frames.shape[1]:
                    reanalysis_frames[batch_index, available_frames:] = (
                        reanalysis_frames[batch_index, available_frames - 1]
                    )

            if action_count < unroll_steps:
                # V2 pads unavailable Atari actions uniformly at random.
                actions[batch_index, action_count:, 0] = self._rng.integers(
                    trajectory.policy_targets.shape[1], size=unroll_steps - action_count
                )
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

            policy_count = min(state_count, block_count - start)
            if policy_count < state_count:
                policy_targets[batch_index].fill(0)
            policy_targets[batch_index, :policy_count] = trajectory.policy_targets[
                start : start + policy_count
            ]
            reanalysis_state_ids[batch_index].fill(-1)
            reanalysis_state_ids[batch_index, :policy_count] = (
                trajectory_state_ids[start : start + policy_count]
            )

            # Only the real/temporary endpoint receives zero-value supervision;
            # later padded states have no prediction loss.
            value_targets[batch_index].fill(0)
            value_targets[batch_index, :policy_count] = trajectory.value_targets[
                start : start + policy_count
            ]

            if value_bootstrap_frames is not None:
                assert value_bootstrap_values is not None
                assert value_bootstrap_discounts is not None
                assert value_bootstrap_mask is not None
                assert value_bootstrap_state_ids is not None
                bootstrap_start = start + self.unroll_steps
                bootstrap_count = min(
                    state_count, max(0, block_count - bootstrap_start)
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
    def _trajectory_key(
        trajectory: GameTrajectory | _StoredTrajectory,
    ) -> tuple[int, int, int]:
        return (
            trajectory.environment_index,
            trajectory.episode_id,
            trajectory.block_id,
        )


__all__ = ["FIFOReplayBuffer", "ReplayAddResult", "ReplayBatch"]
