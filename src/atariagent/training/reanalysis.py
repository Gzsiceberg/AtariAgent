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
    set_mcts_num_threads,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig

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
        mcts_config: MCTSConfig,
        policy_chunk_size: int,
        cache_targets: bool,
        rng_seed: int,
        support_min: int,
        support_max: int,
        precision: Precision,
        mcts_threads: int,
        prefetch_batches: int,
        timeout_seconds: float,
        target_update_interval: int,
        device: torch.device | str | None = None,
    ) -> None:
        if isinstance(cache_targets, bool) is False:
            raise TypeError("cache_targets must be a boolean")
        if isinstance(mcts_threads, bool) or not isinstance(mcts_threads, int):
            raise TypeError("mcts_threads must be an integer")
        if mcts_threads <= 0:
            raise ValueError("mcts_threads must be positive")
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
        set_mcts_num_threads(mcts_threads)

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
            mcts_config,
            seed=rng_seed,
            support_min=support_min,
            support_max=support_max,
            precision=precision,
            chunk_size=policy_chunk_size,
        )
        self.prefetch_batches = prefetch_batches
        self.timeout_seconds = timeout_seconds
        self.target_update_interval = target_update_interval
        self._engine = NativeReanalysisEngine(
            self.target,
            str(self.device),
            prefetch_batches,
            timeout_seconds,
            target_update_interval,
            cache_targets,
        )
        self._latest_target_state: TargetState | None = None
        self._closed = False
        self.max_observed_pending = 0
        self.max_observed_pending_bytes = 0
        self._pending_bytes: dict[int, int] = {}

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

    def submit(self, batch: ReplayBatch) -> int:
        """Queue one ReplayBatch by retaining its existing tensor storage."""
        self._require_open()
        request_id = int(self._engine.submit(batch))
        self._pending_bytes[request_id] = replay_batch_nbytes(batch)
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
        return ReadyReanalysis(
            request_id=request_id,
            batch=result["batch"],
            weight_version=int(result["weight_version"]),
            queue_wait_ms=float(result["queue_wait_ms"]),
            worker_duration_ms=float(result["worker_duration_ms"]),
            transfer_duration_ms=0.0,
            peak_memory_bytes=int(result["peak_memory_bytes"]),
            policy_roots_requested=int(result["policy_roots_requested"]),
            policy_roots_searched=int(result["policy_roots_searched"]),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_bytes.clear()
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
