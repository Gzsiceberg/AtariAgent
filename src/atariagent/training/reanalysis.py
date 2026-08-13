"""Single-actor asynchronous target reanalysis using NumPy Ray payloads."""

from __future__ import annotations

from dataclasses import dataclass, fields
import logging
import math
from time import perf_counter
from typing import Any, Mapping
import warnings

import numpy as np
import ray
import torch
from torch import Tensor, nn

from atariagent.models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig
from atariagent.search._mcts_native import set_num_threads
from .target import Precision, ValueTargetNetwork


TargetState = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class ReanalysisRequest:
    """Target-only NumPy payload passed directly through Ray's object store."""

    request_id: int
    weight_version: int
    indices: np.ndarray
    frames: np.ndarray
    policy_targets: np.ndarray
    value_targets: np.ndarray
    policy_mask: np.ndarray
    value_bootstrap_frames: np.ndarray | None
    value_bootstrap_values: np.ndarray | None
    value_bootstrap_discounts: np.ndarray | None
    value_bootstrap_mask: np.ndarray | None

    @property
    def batch_size(self) -> int:
        return int(self.frames.shape[0])

    @property
    def payload_bytes(self) -> int:
        return sum(
            value.nbytes
            for field in fields(self)
            for value in (getattr(self, field.name),)
            if isinstance(value, np.ndarray)
        )


@dataclass(frozen=True, slots=True)
class ReanalysisResult:
    """Small target-only NumPy result returned by the reanalysis actor."""

    request_id: int
    weight_version: int
    value_targets: np.ndarray
    policy_targets: np.ndarray
    actor_duration_ms: float
    transfer_duration_ms: float
    peak_memory_bytes: int
    policy_roots_requested: int
    policy_roots_searched: int
    cache_size: int


@dataclass(frozen=True, slots=True)
class ReadyReanalysis:
    """A driver-owned batch merged with its asynchronous target result."""

    request_id: int
    batch: ReplayBatch
    weight_version: int
    queue_wait_ms: float
    actor_duration_ms: float
    transfer_duration_ms: float
    peak_memory_bytes: int
    policy_roots_requested: int = 0
    policy_roots_searched: int = 0


@dataclass(slots=True)
class _PendingRequest:
    request_id: int
    batch: ReplayBatch
    submitted_at: float
    payload_bytes: int


def replay_batch_nbytes(batch: ReplayBatch) -> int:
    """Return the tensor payload bytes retained for one replay batch."""
    return sum(
        value.numel() * value.element_size()
        for field in fields(batch)
        for value in (getattr(batch, field.name),)
        if isinstance(value, Tensor)
    )


def make_target_state(
    representation: nn.Module,
    prediction: nn.Module,
    dynamics: nn.Module | None,
) -> TargetState:
    """Copy online network state to one flat CPU target-state dictionary."""
    state: TargetState = {}
    modules = (
        ("representation", representation),
        ("prediction", prediction),
        ("dynamics", dynamics),
    )
    for prefix, module in modules:
        if module is None:
            continue
        for name, value in module.state_dict().items():
            state[f"{prefix}.{name}"] = value.detach().cpu().clone()
    return state


def _numpy_tensor(array: np.ndarray) -> Tensor:
    """Create a read-only tensor view over a Ray NumPy object-store array."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
        )
        return torch.from_numpy(array)


class ReanalysisWorker:
    """One Ray actor owning the target model, MCTS, and target cache."""

    def __init__(
        self,
        *,
        in_channels: int,
        action_space_size: int,
        mcts_config: MCTSConfig,
        policy_chunk_size: int,
        cache_targets: bool,
        rng_seed: int,
        support_min: int,
        support_max: int,
        precision: Precision,
        mcts_threads: int,
    ) -> None:
        if not isinstance(cache_targets, bool):
            raise TypeError("cache_targets must be a boolean")
        for value, name in (
            (policy_chunk_size, "policy_chunk_size"),
            (mcts_threads, "mcts_threads"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        set_num_threads(mcts_threads)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        representation = RepresentationNetwork(in_channels)
        prediction = PredictionNetwork(action_space_size)
        dynamics = DynamicsNetwork(action_space_size)
        self.target = ValueTargetNetwork(
            representation,
            prediction,
            dynamics=dynamics,
            action_space_size=action_space_size,
            mcts_config=mcts_config,
            rng_seed=rng_seed,
            support_min=support_min,
            support_max=support_max,
            precision=precision,
        ).to(self.device)
        self.policy_chunk_size = policy_chunk_size
        self.cache_targets = cache_targets
        self.weight_version = -1
        self._last_request_id = -1
        self._target_cache: dict[int, tuple[Tensor, Tensor]] = {}

    def set_weights(
        self,
        version: int,
        state: Mapping[str, np.ndarray],
    ) -> int:
        """Load target weights and clear targets from the previous version."""
        if version <= self.weight_version:
            raise ValueError("target weight version must increase")
        tensor_state = {name: _numpy_tensor(value) for name, value in state.items()}
        self.target.load_state_dict(tensor_state, strict=True)
        self.target.eval()
        self._target_cache.clear()
        self.weight_version = version
        return version

    def reanalyze(self, request: ReanalysisRequest) -> ReanalysisResult:
        """Resolve cache hits and reanalyze misses for one replay batch."""
        if request.request_id <= self._last_request_id:
            raise ValueError("reanalysis request IDs must increase")
        if request.weight_version != self.weight_version:
            raise ValueError(
                "request target version does not match actor target version"
            )
        self._last_request_id = request.request_id

        batch = self._batch_from_request(request)
        policy_roots_requested = int(batch.policy_mask.sum().item())
        prepared, cache_misses = self._prepare_cache_request(batch)
        policy_roots_searched = int(prepared.policy_mask.sum().item())

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = perf_counter()
        transfer_duration_ms = 0.0
        if policy_roots_searched:
            transfer_started = perf_counter()
            device_batch = prepared.to_reanalysis_device(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            transfer_duration_ms = (
                perf_counter() - transfer_started
            ) * 1_000.0
            reanalyzed = self.target.reanalyze_batch(
                device_batch,
                policy_chunk_size=self.policy_chunk_size,
            )
            output_transfer_started = perf_counter()
            value_targets = (
                reanalyzed.value_targets.detach().cpu().contiguous()
            )
            policy_targets = (
                reanalyzed.policy_targets.detach().cpu().contiguous()
            )
            transfer_duration_ms += (
                perf_counter() - output_transfer_started
            ) * 1_000.0
        else:
            # A fully cached request needs neither CUDA transfer nor inference.
            value_targets = prepared.value_targets.contiguous()
            policy_targets = prepared.policy_targets.contiguous()
        value_targets, policy_targets = self._resolve_cache_misses(
            value_targets,
            policy_targets,
            cache_misses,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
        else:
            peak_memory_bytes = 0
        return ReanalysisResult(
            request_id=request.request_id,
            weight_version=request.weight_version,
            value_targets=value_targets.numpy(),
            policy_targets=policy_targets.numpy(),
            actor_duration_ms=(perf_counter() - started) * 1_000.0,
            transfer_duration_ms=transfer_duration_ms,
            peak_memory_bytes=peak_memory_bytes,
            policy_roots_requested=policy_roots_requested,
            policy_roots_searched=policy_roots_searched,
            cache_size=len(self._target_cache),
        )

    def _prepare_cache_request(
        self,
        batch: ReplayBatch,
    ) -> tuple[
        ReplayBatch,
        dict[int, tuple[tuple[int, int], ...]],
    ]:
        """Merge cache hits and select one search position per missing ID."""
        if not self.cache_targets:
            return batch, {}

        positions_by_id: dict[int, list[tuple[int, int]]] = {}
        for sample, offset in batch.policy_mask.nonzero().tolist():
            state_id = int(batch.indices[sample]) + offset
            positions_by_id.setdefault(state_id, []).append((sample, offset))

        value_targets = batch.value_targets.clone()
        policy_targets = batch.policy_targets.clone()
        miss_mask = torch.zeros_like(batch.policy_mask)
        cache_misses: dict[int, tuple[tuple[int, int], ...]] = {}
        for state_id, position_list in positions_by_id.items():
            positions = tuple(position_list)
            cached = self._target_cache.get(state_id)
            if cached is None:
                miss_mask[positions[0]] = True
                cache_misses[state_id] = positions
                continue
            value, policy = cached
            for position in positions:
                value_targets[position] = value
                policy_targets[position] = policy

        bootstrap_mask = batch.value_bootstrap_mask
        if bootstrap_mask is not None:
            bootstrap_mask = bootstrap_mask & miss_mask
        return (
            ReplayBatch(
                frames=batch.frames,
                actions=batch.actions,
                rewards=batch.rewards,
                policy_targets=policy_targets,
                value_targets=value_targets,
                action_mask=batch.action_mask,
                policy_mask=batch.policy_mask & miss_mask,
                value_mask=batch.value_mask,
                indices=batch.indices,
                importance_weights=batch.importance_weights,
                value_bootstrap_frames=batch.value_bootstrap_frames,
                value_bootstrap_values=batch.value_bootstrap_values,
                value_bootstrap_discounts=batch.value_bootstrap_discounts,
                value_bootstrap_mask=bootstrap_mask,
            ),
            cache_misses,
        )

    def _resolve_cache_misses(
        self,
        value_targets: Tensor,
        policy_targets: Tensor,
        cache_misses: dict[int, tuple[tuple[int, int], ...]],
    ) -> tuple[Tensor, Tensor]:
        """Expand unique searched targets and populate the actor cache."""
        for state_id, positions in cache_misses.items():
            source = positions[0]
            value = value_targets[source].clone()
            policy = policy_targets[source].clone()
            for position in positions[1:]:
                value_targets[position] = value
                policy_targets[position] = policy
            self._target_cache[state_id] = (value, policy)
        return value_targets, policy_targets

    @staticmethod
    def _batch_from_request(request: ReanalysisRequest) -> ReplayBatch:
        """Construct the target-only batch view consumed by the actor."""
        unroll_steps = request.policy_targets.shape[1] - 1
        batch_size = request.batch_size

        def optional(array: np.ndarray | None) -> Tensor | None:
            return None if array is None else _numpy_tensor(array)

        value_targets = _numpy_tensor(request.value_targets)
        return ReplayBatch(
            frames=_numpy_tensor(request.frames),
            actions=torch.zeros(batch_size, unroll_steps, 1, dtype=torch.long),
            rewards=torch.zeros(batch_size, unroll_steps),
            policy_targets=_numpy_tensor(request.policy_targets),
            value_targets=value_targets,
            action_mask=torch.ones(batch_size, unroll_steps, dtype=torch.bool),
            policy_mask=_numpy_tensor(request.policy_mask),
            value_mask=torch.ones_like(value_targets, dtype=torch.bool),
            indices=_numpy_tensor(request.indices),
            importance_weights=torch.ones(batch_size),
            value_bootstrap_frames=optional(request.value_bootstrap_frames),
            value_bootstrap_values=optional(request.value_bootstrap_values),
            value_bootstrap_discounts=optional(
                request.value_bootstrap_discounts
            ),
            value_bootstrap_mask=optional(request.value_bootstrap_mask),
        )


class ReanalysisPipeline:
    """Bounded request submission and result matching for one actor."""

    def __init__(
        self,
        actor: Any,
        *,
        prefetch_batches: int,
        timeout_seconds: float,
        max_weight_lag: int,
        ray_api: Any = ray,
    ) -> None:
        if isinstance(prefetch_batches, bool) or not isinstance(
            prefetch_batches, int
        ):
            raise TypeError("prefetch_batches must be an integer")
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if isinstance(max_weight_lag, bool) or not isinstance(max_weight_lag, int):
            raise TypeError("max_weight_lag must be an integer")
        if max_weight_lag < 0:
            raise ValueError("max_weight_lag must be non-negative")

        self.actor = actor
        self.prefetch_batches = prefetch_batches
        self.timeout_seconds = timeout_seconds
        self.max_weight_lag = max_weight_lag
        self._ray = ray_api
        self._inflight: dict[Any, _PendingRequest] = {}
        self._next_request_id = 0
        self._weight_version = -1
        self._latest_target_state: TargetState | None = None
        self._cache_size = 0
        self._closed = False
        self.max_observed_pending = 0
        self.max_observed_pending_bytes = 0

    @property
    def pending_count(self) -> int:
        return len(self._inflight)

    @property
    def pending_payload_bytes(self) -> int:
        return sum(
            request.payload_bytes for request in self._inflight.values()
        )

    @property
    def needs_prefetch(self) -> bool:
        return self.pending_count < self.prefetch_batches

    @property
    def cache_size(self) -> int:
        """Return the cache size reported by the latest actor result."""
        return self._cache_size

    @property
    def weight_version(self) -> int:
        return self._weight_version

    @property
    def latest_target_state(self) -> TargetState | None:
        return self._latest_target_state

    def publish_weights(
        self,
        version: int,
        state: TargetState,
        *,
        wait: bool = False,
    ) -> None:
        """Queue a NumPy weight snapshot after older actor requests."""
        self._require_open()
        if version <= self._weight_version:
            raise ValueError("target weight version must increase")
        numpy_state = {
            name: value.detach().cpu().contiguous().numpy()
            for name, value in state.items()
        }
        update_ref = self.actor.set_weights.remote(version, numpy_state)
        if wait:
            loaded_version = self._ray.get(update_ref)
            if loaded_version != version:
                raise RuntimeError("actor acknowledged an invalid weight version")
        self._latest_target_state = state
        self._weight_version = version
        self._cache_size = 0

    def submit(self, batch: ReplayBatch) -> int:
        """Submit one replay batch as a direct NumPy actor payload."""
        self._require_open()
        if self._weight_version < 0:
            raise RuntimeError("target weights must be published before submission")
        if not self.needs_prefetch:
            raise BufferError("reanalysis prefetch limit reached")
        request_id = self._next_request_id
        self._next_request_id += 1
        request = self._request_from_batch(
            request_id,
            self._weight_version,
            batch,
        )
        result_ref = self.actor.reanalyze.remote(request)
        self._inflight[result_ref] = _PendingRequest(
            request_id=request_id,
            batch=batch,
            submitted_at=perf_counter(),
            payload_bytes=request.payload_bytes,
        )
        self.max_observed_pending = max(
            self.max_observed_pending,
            self.pending_count,
        )
        self.max_observed_pending_bytes = max(
            self.max_observed_pending_bytes,
            self.pending_payload_bytes,
        )
        return request_id

    def wait_next(self) -> ReadyReanalysis:
        """Wait for one actor result and merge it with its retained batch."""
        self._require_open()
        if not self.pending_count:
            raise RuntimeError("no pending reanalysis batches")
        ready_refs, _ = self._ray.wait(
            list(self._inflight),
            num_returns=1,
            timeout=self.timeout_seconds,
        )
        if not ready_refs:
            raise TimeoutError("timed out waiting for asynchronous reanalysis")
        result_ref = ready_refs[0]
        pending = self._inflight.pop(result_ref)
        result = self._ray.get(result_ref)
        if not isinstance(result, ReanalysisResult):
            raise TypeError("reanalysis actor returned an invalid result")
        if result.request_id != pending.request_id:
            raise RuntimeError("reanalysis result request ID mismatch")
        lag = self._weight_version - result.weight_version
        if lag < 0 or lag > self.max_weight_lag:
            raise RuntimeError(
                f"reanalysis result target-weight lag {lag} exceeds limit"
            )
        self._cache_size = result.cache_size
        value_targets = torch.from_numpy(result.value_targets.copy())
        policy_targets = torch.from_numpy(result.policy_targets.copy())
        if pending.batch.frames.is_pinned():
            value_targets = value_targets.pin_memory()
            policy_targets = policy_targets.pin_memory()
        batch = pending.batch.with_reanalysis_targets(
            value_targets=value_targets,
            policy_targets=policy_targets,
        )
        return ReadyReanalysis(
            request_id=result.request_id,
            batch=batch,
            weight_version=result.weight_version,
            queue_wait_ms=(perf_counter() - pending.submitted_at) * 1_000.0,
            actor_duration_ms=result.actor_duration_ms,
            transfer_duration_ms=result.transfer_duration_ms,
            peak_memory_bytes=result.peak_memory_bytes,
            policy_roots_requested=result.policy_roots_requested,
            policy_roots_searched=result.policy_roots_searched,
        )

    def close(self) -> None:
        """Terminate the actor and release retained batches and object refs."""
        if self._closed:
            return
        self._closed = True
        self._inflight.clear()
        self._ray.kill(self.actor, no_restart=True)

    @staticmethod
    def _request_from_batch(
        request_id: int,
        weight_version: int,
        batch: ReplayBatch,
    ) -> ReanalysisRequest:
        def numpy(name: str) -> np.ndarray:
            value = getattr(batch, name)
            if not isinstance(value, Tensor) or value.device.type != "cpu":
                raise ValueError("reanalysis payload tensors must be on CPU")
            return value.detach().contiguous().numpy()

        def optional_numpy(name: str) -> np.ndarray | None:
            value = getattr(batch, name)
            if value is None:
                return None
            if value.device.type != "cpu":
                raise ValueError("reanalysis payload tensors must be on CPU")
            return value.detach().contiguous().numpy()

        return ReanalysisRequest(
            request_id=request_id,
            weight_version=weight_version,
            indices=numpy("indices"),
            frames=numpy("frames"),
            policy_targets=numpy("policy_targets"),
            value_targets=numpy("value_targets"),
            policy_mask=numpy("policy_mask"),
            value_bootstrap_frames=optional_numpy("value_bootstrap_frames"),
            value_bootstrap_values=optional_numpy("value_bootstrap_values"),
            value_bootstrap_discounts=optional_numpy(
                "value_bootstrap_discounts"
            ),
            value_bootstrap_mask=optional_numpy("value_bootstrap_mask"),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("reanalysis pipeline is closed")


def create_reanalysis_actor(
    *,
    num_gpus: float,
    num_cpus: int = 1,
    mcts_threads: int = 1,
    in_channels: int,
    action_space_size: int,
    mcts_config: MCTSConfig,
    policy_chunk_size: int,
    cache_targets: bool,
    rng_seed: int,
    support_min: int,
    support_max: int,
    precision: Precision,
) -> Any:
    """Create the local reanalysis actor with explicit resource reservation."""
    if not math.isfinite(num_gpus) or num_gpus < 0.0:
        raise ValueError("reanalysis actor num_gpus must be non-negative")
    for value, name in (
        (num_cpus, "num_cpus"),
        (mcts_threads, "mcts_threads"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    actor_class = ray.remote(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )(ReanalysisWorker)
    return actor_class.remote(
        in_channels=in_channels,
        action_space_size=action_space_size,
        mcts_config=mcts_config,
        policy_chunk_size=policy_chunk_size,
        cache_targets=cache_targets,
        rng_seed=rng_seed,
        support_min=support_min,
        support_max=support_max,
        precision=precision,
        mcts_threads=mcts_threads,
    )


def initialize_local_ray(*, object_store_memory: int | None = None) -> bool:
    """Start a quiet local Ray runtime and report whether this call owns it."""
    if ray.is_initialized():
        return False
    kwargs: dict[str, Any] = {
        "log_to_driver": False,
        "include_dashboard": False,
        "logging_level": logging.ERROR,
    }
    if object_store_memory is not None:
        if object_store_memory <= 0:
            raise ValueError("object_store_memory must be positive")
        kwargs["object_store_memory"] = object_store_memory
    ray.init(**kwargs)
    return True


__all__ = [
    "ReadyReanalysis",
    "ReanalysisPipeline",
    "ReanalysisRequest",
    "ReanalysisResult",
    "ReanalysisWorker",
    "TargetState",
    "create_reanalysis_actor",
    "initialize_local_ray",
    "make_target_state",
    "replay_batch_nbytes",
]
