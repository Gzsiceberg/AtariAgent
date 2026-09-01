"""In-process C++ threaded target reanalysis without Ray transport."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal

import torch
from torch import Tensor, nn

from atariagent.models.native import (
    DynamicsNetwork,
    InferenceModels,
    NativeReanalysisEngine,
    PredictionNetwork,
    RepresentationNetwork,
    make_value_target,
    set_tree_search_num_threads,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.search import SearchConfig

from .config import puct_root_noise_temperature

Precision = Literal["fp32", "bf16"]
TargetState = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class ReadyReanalysis:
    """A driver-owned batch completed by the native reanalysis thread."""

    request_id: int
    batch: ReplayBatch
    weight_version: int
    queue_wait_ms: float
    worker_duration_ms: float
    transfer_duration_ms: float
    peak_memory_bytes: int
    policy_roots_requested: int = 0
    policy_roots_searched: int = 0
    cache_hits: int = 0
    cache_target_age_mean: float = 0.0
    cache_target_age_max: int = 0


def replay_batch_nbytes(batch: ReplayBatch) -> int:
    """Return unique tensor storage retained by one queued replay batch."""
    storages: dict[tuple[str, int | None, int], int] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not isinstance(value, Tensor):
            continue
        storage = value.untyped_storage()
        key = (value.device.type, value.device.index, storage.data_ptr())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def make_target_state(
    representation: nn.Module,
    prediction: nn.Module,
    dynamics: nn.Module | None,
) -> TargetState:
    """Copy online module state for checkpoints and target publication."""
    state: TargetState = {}
    for prefix, module in (
        ("representation", representation),
        ("prediction", prediction),
        ("dynamics", dynamics),
    ):
        if module is None:
            continue
        for name, value in module.state_dict().items():
            state[f"{prefix}.{name}"] = value.detach().cpu().clone()
    return state


def _state_section(state: TargetState, prefix: str) -> dict[str, Tensor]:
    """Return one network's unprefixed tensor state."""
    marker = f"{prefix}."
    return {
        name.removeprefix(marker): value
        for name, value in state.items()
        if name.startswith(marker)
    }


class ReanalysisPipeline:
    """Bounded native-thread reanalysis pipeline with zero-copy requests."""

    def __init__(
        self,
        *,
        in_channels: int,
        action_space_size: int,
        search_config: SearchConfig,
        policy_chunk_size: int,
        cache_targets: bool,
        cache_target_ttl: int,
        policy_reanalysis_ramp_transitions: int,
        rng_seed: int,
        support_min: int,
        support_max: int,
        precision: Precision,
        search_threads: int,
        prefetch_batches: int,
        timeout_seconds: float,
        target_update_interval: int,
        root_noise_total_steps: int,
        collection_steps: int,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(cache_targets, bool):
            raise TypeError("cache_targets must be a boolean")
        if isinstance(cache_target_ttl, bool) or not isinstance(
            cache_target_ttl, int
        ):
            raise TypeError("cache_target_ttl must be an integer")
        if cache_target_ttl < 0:
            raise ValueError("cache_target_ttl must be non-negative")
        if isinstance(policy_reanalysis_ramp_transitions, bool) or not isinstance(
            policy_reanalysis_ramp_transitions, int
        ):
            raise TypeError(
                "policy_reanalysis_ramp_transitions must be an integer"
            )
        if policy_reanalysis_ramp_transitions <= 0:
            raise ValueError(
                "policy_reanalysis_ramp_transitions must be positive"
            )
        if isinstance(search_threads, bool) or not isinstance(search_threads, int):
            raise TypeError("search_threads must be an integer")
        if search_threads <= 0:
            raise ValueError("search_threads must be positive")
        if isinstance(prefetch_batches, bool) or not isinstance(prefetch_batches, int):
            raise TypeError("prefetch_batches must be an integer")
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if isinstance(target_update_interval, bool) or not isinstance(
            target_update_interval, int
        ):
            raise TypeError("target_update_interval must be an integer")
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if isinstance(root_noise_total_steps, bool) or not isinstance(
            root_noise_total_steps, int
        ):
            raise TypeError("root_noise_total_steps must be an integer")
        if root_noise_total_steps <= 0:
            raise ValueError("root_noise_total_steps must be positive")
        if isinstance(collection_steps, bool) or not isinstance(
            collection_steps, int
        ):
            raise TypeError("collection_steps must be an integer")
        if collection_steps <= 0:
            raise ValueError("collection_steps must be positive")
        if collection_steps > root_noise_total_steps:
            raise ValueError("collection_steps must not exceed total steps")
        set_tree_search_num_threads(search_threads)

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        models = (
            InferenceModels(
                representation=RepresentationNetwork(in_channels),
                prediction=PredictionNetwork(action_space_size),
                dynamics=DynamicsNetwork(action_space_size),
            )
            .eval()
            .to(self.device)
        )
        self.target = make_value_target(
            models,
            action_space_size,
            search_config,
            seed=rng_seed,
            support_min=support_min,
            support_max=support_max,
            precision=precision,
            chunk_size=policy_chunk_size,
        )
        self.prefetch_batches = prefetch_batches
        self.timeout_seconds = timeout_seconds
        self.target_update_interval = target_update_interval
        self.root_noise_total_steps = root_noise_total_steps
        self.collection_steps = collection_steps
        self.policy_reanalysis_ramp_transitions = (
            policy_reanalysis_ramp_transitions
        )
        self.search_algorithm = search_config.search_algorithm
        self._engine = NativeReanalysisEngine(
            self.target,
            str(self.device),
            prefetch_batches,
            timeout_seconds,
            target_update_interval,
            cache_targets,
            cache_target_ttl,
        )
        self._latest_target_state: TargetState | None = None
        self._closed = False
        self.max_observed_pending = 0
        self.max_observed_pending_bytes = 0
        self._pending_bytes: dict[int, int] = {}
        self._pending_policy_targets: dict[int, tuple[Tensor, Tensor]] = {}

    @property
    def pending_count(self) -> int:
        return int(self._engine.pending_count)

    @property
    def pending_payload_bytes(self) -> int:
        return sum(self._pending_bytes.values())

    @property
    def needs_prefetch(self) -> bool:
        return self.pending_count < self.prefetch_batches

    @property
    def cache_size(self) -> int:
        return int(self._engine.cache_size)

    @property
    def weight_version(self) -> int:
        return int(self._engine.weight_version)

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
        """Queue an ordered immutable target snapshot and wait for activation."""
        del wait  # Native publication is deliberately synchronous for ordering.
        self._require_open()
        self._engine.publish_weights(
            version,
            _state_section(state, "representation"),
            _state_section(state, "prediction"),
            _state_section(state, "dynamics"),
        )
        self._latest_target_state = dict(state)

    def clear_cache(self) -> None:
        """Queue a cache clear after all previously submitted requests."""
        self._require_open()
        self._engine.clear_cache()

    def root_noise_temperature(self, trained_steps: int) -> float:
        """Return the search-noise temperature for a learner step."""
        if isinstance(trained_steps, bool) or not isinstance(trained_steps, int):
            raise TypeError("trained_steps must be an integer")
        if trained_steps < 0:
            raise ValueError("trained_steps must be non-negative")
        if self.search_algorithm == "gumbel":
            return 0.0
        return puct_root_noise_temperature(
            trained_steps,
            self.root_noise_total_steps,
        )

    def policy_reanalysis_weights(
        self,
        batch: ReplayBatch,
        *,
        trained_steps: int,
    ) -> Tensor:
        """Return search weights from effective replay-transition ages."""
        effective_ages = batch.effective_transition_ages(
            learner_step=trained_steps,
            collection_steps=self.collection_steps,
        )
        return (
            effective_ages.to(dtype=batch.policy_targets.dtype)
            .div(self.policy_reanalysis_ramp_transitions)
            .clamp(max=1.0)
            .reshape(batch.batch_size, 1, 1)
        )

    def submit(self, batch: ReplayBatch, *, trained_steps: int = 0) -> int:
        """Queue one ReplayBatch with schedules evaluated at submission."""
        self._require_open()
        root_noise_temperature = self.root_noise_temperature(trained_steps)
        policy_reanalysis_weights = self.policy_reanalysis_weights(
            batch,
            trained_steps=trained_steps,
        )
        request_id = int(
            self._engine.submit(
                batch,
                root_noise_temperature,
                self.search_algorithm == "gumbel",
                trained_steps,
            )
        )
        self._pending_bytes[request_id] = replay_batch_nbytes(batch)
        self._pending_policy_targets[request_id] = (
            batch.policy_targets,
            policy_reanalysis_weights,
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
        """Wait for one completed native request and return its zero-copy batch."""
        self._require_open()
        result = self._engine.wait_next()
        request_id = int(result["request_id"])
        self._pending_bytes.pop(request_id, None)
        original_policy_targets, search_weights = (
            self._pending_policy_targets.pop(request_id)
        )
        batch = result["batch"]
        blended_policy_targets = torch.lerp(
            original_policy_targets,
            batch.policy_targets,
            search_weights,
        )
        batch = batch.with_reanalysis_targets(
            value_targets=batch.value_targets,
            policy_targets=blended_policy_targets,
            search_value_targets=batch.search_value_targets,
        )
        return ReadyReanalysis(
            request_id=request_id,
            batch=batch,
            weight_version=int(result["weight_version"]),
            queue_wait_ms=float(result["queue_wait_ms"]),
            worker_duration_ms=float(result["worker_duration_ms"]),
            transfer_duration_ms=0.0,
            peak_memory_bytes=int(result["peak_memory_bytes"]),
            policy_roots_requested=int(result["policy_roots_requested"]),
            policy_roots_searched=int(result["policy_roots_searched"]),
            cache_hits=int(result["cache_hits"]),
            cache_target_age_mean=float(result["cache_target_age_mean"]),
            cache_target_age_max=int(result["cache_target_age_max"]),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_bytes.clear()
        self._pending_policy_targets.clear()
        self._engine.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("reanalysis pipeline is closed")


__all__ = [
    "ReadyReanalysis",
    "ReanalysisPipeline",
    "TargetState",
    "make_target_state",
    "replay_batch_nbytes",
]
