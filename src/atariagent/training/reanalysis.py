"""Single-machine asynchronous target reanalysis using local Ray actors."""

from __future__ import annotations

from dataclasses import dataclass, fields
import logging
import math
from time import perf_counter
from typing import Any, Mapping

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
    """One immutable CPU replay batch submitted to the target actor."""

    request_id: int
    weight_version: int
    batch: ReplayBatch
    reanalyze_values: bool
    policy_ratio: float
    policy_chunk_size: int


@dataclass(frozen=True, slots=True)
class ReanalysisResult:
    """Small target-only result returned by the reanalysis actor."""

    request_id: int
    weight_version: int
    value_targets: Tensor
    policy_targets: Tensor
    actor_duration_ms: float
    transfer_duration_ms: float
    peak_memory_bytes: int


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


@dataclass(slots=True)
class _PendingRequest:
    request_id: int
    actor_index: int
    batch: ReplayBatch
    request_ref: Any
    submitted_at: float
    payload_bytes: int


def replay_batch_nbytes(batch: ReplayBatch) -> int:
    """Return the tensor payload bytes retained for one pending replay batch."""
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


class ReanalysisWorker:
    """Ray actor implementation owning the frozen target model and MCTS."""

    def __init__(
        self,
        *,
        in_channels: int,
        action_space_size: int,
        mcts_config: MCTSConfig,
        policy_enabled: bool,
        rng_seed: int,
        support_min: int,
        support_max: int,
        precision: Precision,
        mcts_threads: int,
    ) -> None:
        if isinstance(mcts_threads, bool) or not isinstance(mcts_threads, int):
            raise TypeError("mcts_threads must be an integer")
        if mcts_threads <= 0:
            raise ValueError("mcts_threads must be positive")
        set_num_threads(mcts_threads)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        representation = RepresentationNetwork(in_channels)
        prediction = PredictionNetwork(action_space_size)
        dynamics = (
            DynamicsNetwork(action_space_size) if policy_enabled else None
        )
        self.target = ValueTargetNetwork(
            representation,
            prediction,
            dynamics=dynamics,
            action_space_size=(action_space_size if policy_enabled else None),
            mcts_config=mcts_config,
            rng_seed=rng_seed,
            support_min=support_min,
            support_max=support_max,
            precision=precision,
        ).to(self.device)
        self.weight_version = -1
        self._last_request_id = -1

    def set_weights(
        self,
        version: int,
        state: Mapping[str, Tensor],
    ) -> int:
        """Load a monotonically versioned CPU target snapshot."""
        if version <= self.weight_version:
            raise ValueError("target weight version must increase")
        self.target.load_state_dict(state, strict=True)
        self.target.eval()
        self.weight_version = version
        return version

    def reanalyze(self, request: ReanalysisRequest) -> ReanalysisResult:
        """Run target inference/MCTS and return only corrected target tensors."""
        if request.request_id <= self._last_request_id:
            raise ValueError("reanalysis request IDs must increase")
        if request.weight_version != self.weight_version:
            raise ValueError(
                "request target version does not match actor target version"
            )
        self._last_request_id = request.request_id

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = perf_counter()
        transfer_started = perf_counter()
        batch = request.batch.to_reanalysis_device(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        transfer_duration_ms = (perf_counter() - transfer_started) * 1_000.0

        reanalyzed = self.target.reanalyze_batch(
            batch,
            reanalyze_values=request.reanalyze_values,
            policy_ratio=request.policy_ratio,
            policy_chunk_size=request.policy_chunk_size,
        )
        output_transfer_started = perf_counter()
        value_targets = reanalyzed.value_targets.detach().cpu().contiguous()
        policy_targets = reanalyzed.policy_targets.detach().cpu().contiguous()
        transfer_duration_ms += (
            perf_counter() - output_transfer_started
        ) * 1_000.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
        else:
            peak_memory_bytes = 0
        actor_duration_ms = (perf_counter() - started) * 1_000.0
        return ReanalysisResult(
            request_id=request.request_id,
            weight_version=request.weight_version,
            value_targets=value_targets,
            policy_targets=policy_targets,
            actor_duration_ms=actor_duration_ms,
            transfer_duration_ms=transfer_duration_ms,
            peak_memory_bytes=peak_memory_bytes,
        )


class ReanalysisPipeline:
    """Driver-side bounded submission, result matching, and backpressure."""

    def __init__(
        self,
        actors: Any,
        *,
        reanalyze_values: bool,
        policy_ratio: float,
        policy_chunk_size: int,
        prefetch_batches: int,
        timeout_seconds: float,
        max_weight_lag: int,
        ray_api: Any = ray,
    ) -> None:
        if not isinstance(reanalyze_values, bool):
            raise TypeError("reanalyze_values must be a boolean")
        if not math.isfinite(policy_ratio) or not 0.0 <= policy_ratio <= 1.0:
            raise ValueError("policy_ratio must be in [0, 1]")
        for value, name in (
            (policy_chunk_size, "policy_chunk_size"),
            (prefetch_batches, "prefetch_batches"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if isinstance(max_weight_lag, bool) or not isinstance(max_weight_lag, int):
            raise TypeError("max_weight_lag must be an integer")
        if max_weight_lag < 0:
            raise ValueError("max_weight_lag must be non-negative")

        self.actors = (
            tuple(actors)
            if isinstance(actors, (list, tuple))
            else (actors,)
        )
        if not self.actors:
            raise ValueError("at least one reanalysis actor is required")
        self.reanalyze_values = reanalyze_values
        self.policy_ratio = policy_ratio
        self.policy_chunk_size = policy_chunk_size
        self.prefetch_batches = prefetch_batches
        self.timeout_seconds = timeout_seconds
        self.max_weight_lag = max_weight_lag
        self._ray = ray_api
        self._pending: dict[Any, _PendingRequest] = {}
        self._next_request_id = 0
        self._next_actor_index = 0
        self._actor_pending_counts = [0] * len(self.actors)
        self._weight_version = -1
        self._latest_target_state: TargetState | None = None
        self._latest_weight_ref: Any = None
        self._closed = False
        self.max_observed_pending = 0
        self.max_observed_pending_bytes = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_payload_bytes(self) -> int:
        return sum(request.payload_bytes for request in self._pending.values())

    @property
    def needs_prefetch(self) -> bool:
        return self.pending_count < self.prefetch_batches

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
        """Publish one snapshot before submitting requests with its version."""
        self._require_open()
        if version <= self._weight_version:
            raise ValueError("target weight version must increase")
        state_ref = self._ray.put(state)
        update_refs = [
            actor.set_weights.remote(version, state_ref)
            for actor in self.actors
        ]
        if wait:
            for update_ref in update_refs:
                loaded_version = self._ray.get(update_ref)
                if loaded_version != version:
                    raise RuntimeError(
                        "actor acknowledged an invalid weight version"
                    )
        self._latest_target_state = state
        self._latest_weight_ref = state_ref
        self._weight_version = version

    def submit(self, batch: ReplayBatch) -> int:
        """Submit one batch unless the prefetch depth is already reached."""
        self._require_open()
        if self._weight_version < 0:
            raise RuntimeError("target weights must be published before submission")
        if self.pending_count >= self.prefetch_batches:
            raise BufferError("reanalysis prefetch limit reached")
        request_id = self._next_request_id
        self._next_request_id += 1
        request = ReanalysisRequest(
            request_id=request_id,
            weight_version=self._weight_version,
            batch=batch,
            reanalyze_values=self.reanalyze_values,
            policy_ratio=self.policy_ratio,
            policy_chunk_size=self.policy_chunk_size,
        )
        actor_index = self._select_actor()
        request_ref = self._ray.put(request)
        result_ref = self.actors[actor_index].reanalyze.remote(request_ref)
        self._actor_pending_counts[actor_index] += 1
        self._pending[result_ref] = _PendingRequest(
            request_id=request_id,
            actor_index=actor_index,
            batch=batch,
            request_ref=request_ref,
            submitted_at=perf_counter(),
            payload_bytes=replay_batch_nbytes(batch),
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
        """Wait for one target result and merge it with the retained CPU batch."""
        self._require_open()
        if not self._pending:
            raise RuntimeError("no pending reanalysis batches")
        ready, _ = self._ray.wait(
            list(self._pending),
            num_returns=1,
            timeout=self.timeout_seconds,
        )
        if not ready:
            raise TimeoutError("timed out waiting for asynchronous reanalysis")
        result_ref = ready[0]
        pending = self._pending.pop(result_ref)
        self._actor_pending_counts[pending.actor_index] -= 1
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
        batch = pending.batch.with_reanalysis_targets(
            value_targets=result.value_targets,
            policy_targets=result.policy_targets,
        )
        return ReadyReanalysis(
            request_id=result.request_id,
            batch=batch,
            weight_version=result.weight_version,
            queue_wait_ms=(perf_counter() - pending.submitted_at) * 1_000.0,
            actor_duration_ms=result.actor_duration_ms,
            transfer_duration_ms=result.transfer_duration_ms,
            peak_memory_bytes=result.peak_memory_bytes,
        )

    def close(self) -> None:
        """Terminate the actor and release all retained batches/object refs."""
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._actor_pending_counts = [0] * len(self.actors)
        self._latest_weight_ref = None
        for actor in self.actors:
            self._ray.kill(actor, no_restart=True)

    def _select_actor(self) -> int:
        minimum_pending = min(self._actor_pending_counts)
        for offset in range(len(self.actors)):
            index = (self._next_actor_index + offset) % len(self.actors)
            if self._actor_pending_counts[index] == minimum_pending:
                self._next_actor_index = (index + 1) % len(self.actors)
                return index
        raise RuntimeError("failed to select a reanalysis actor")

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
    policy_enabled: bool,
    rng_seed: int,
    support_min: int,
    support_max: int,
    precision: Precision,
) -> Any:
    """Create one silent local Ray actor with explicit GPU reservation."""
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
        policy_enabled=policy_enabled,
        rng_seed=rng_seed,
        support_min=support_min,
        support_max=support_max,
        precision=precision,
        mcts_threads=mcts_threads,
    )


def create_reanalysis_actors(
    *,
    count: int,
    num_gpus: float,
    num_cpus: int,
    mcts_threads: int,
    in_channels: int,
    action_space_size: int,
    mcts_config: MCTSConfig,
    policy_enabled: bool,
    rng_seed: int,
    support_min: int,
    support_max: int,
    precision: Precision,
) -> tuple[Any, ...]:
    """Create a local actor pool with independent deterministic RNG streams."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("reanalysis actor count must be an integer")
    if count <= 0:
        raise ValueError("reanalysis actor count must be positive")
    actors: list[Any] = []
    try:
        for index in range(count):
            actors.append(
                create_reanalysis_actor(
                    num_gpus=num_gpus,
                    num_cpus=num_cpus,
                    mcts_threads=mcts_threads,
                    in_channels=in_channels,
                    action_space_size=action_space_size,
                    mcts_config=mcts_config,
                    policy_enabled=policy_enabled,
                    rng_seed=rng_seed + index,
                    support_min=support_min,
                    support_max=support_max,
                    precision=precision,
                )
            )
    except Exception:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        raise
    return tuple(actors)


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
    "create_reanalysis_actors",
    "initialize_local_ray",
    "make_target_state",
    "replay_batch_nbytes",
]
