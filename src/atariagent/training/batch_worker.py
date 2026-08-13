"""Bounded threaded replay sampling and asynchronous CUDA batch transfer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from queue import Empty, Full, Queue, SimpleQueue
from threading import Event, Lock, Semaphore, Thread
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from atariagent.replay import FIFOReplayBuffer
from atariagent.replay_batch import ReplayBatch
from .reanalysis import ReanalysisPipeline, TargetState


@dataclass(frozen=True, slots=True)
class ReadyBatch:
    """One GPU batch whose asynchronous transfer has been enqueued."""

    token: int = field(repr=False)
    sample_step: int
    cpu_batch: ReplayBatch = field(repr=False)
    gpu_batch: ReplayBatch
    priority_beta: float
    ready_event: torch.cuda.Event | None = field(repr=False)
    sample_duration_ms: float
    transfer_enqueue_ms: float
    queue_wait_ms: float | None = None
    actor_duration_ms: float | None = None
    policy_roots_requested: int = 0
    policy_roots_searched: int = 0

    def wait_for_current_stream(self, device: torch.device | str) -> None:
        """Make the current training stream wait for this batch's transfer."""
        resolved = torch.device(device)
        if self.ready_event is None:
            return
        training_stream = torch.cuda.current_stream(resolved)
        training_stream.wait_event(self.ready_event)
        self.gpu_batch.record_stream(training_stream)


@dataclass(frozen=True, slots=True)
class _Run:
    start_step: int
    count: int


@dataclass(slots=True)
class _PublishWeights:
    version: int
    state: TargetState
    done: Event = field(default_factory=Event)
    error: BaseException | None = None


class _StopWorker(Exception):
    pass


class BatchWorker:
    """Prefetch replay batches with bounded backpressure.

    Replay sampling runs in one background thread. Optional target reanalysis is
    submitted and consumed by that same thread, then each pinned CPU batch is
    copied on a dedicated CUDA stream. A global semaphore bounds batches across
    sampling, reanalysis, the ready queue, and learner consumption.
    """

    def __init__(
        self,
        replay: FIFOReplayBuffer,
        *,
        batch_size: int,
        training_steps: int,
        device: torch.device | str,
        reanalysis_pipeline: ReanalysisPipeline | None = None,
        reanalysis_start_step: int = 0,
        max_in_flight: int = 3,
        ready_prefetch: int = 1,
        timeout_seconds: float = 600.0,
    ) -> None:
        for value, name in (
            (batch_size, "batch_size"),
            (training_steps, "training_steps"),
            (reanalysis_start_step, "reanalysis_start_step"),
            (max_in_flight, "max_in_flight"),
            (ready_prefetch, "ready_prefetch"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if training_steps <= 0:
            raise ValueError("training_steps must be positive")
        if reanalysis_start_step < 0:
            raise ValueError("reanalysis_start_step must be non-negative")
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        if ready_prefetch <= 0:
            raise ValueError("ready_prefetch must be positive")
        if ready_prefetch > max_in_flight:
            raise ValueError("ready_prefetch must not exceed max_in_flight")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite")

        self.replay = replay
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.device = torch.device(device)
        self.reanalysis_pipeline = reanalysis_pipeline
        self.reanalysis_start_step = reanalysis_start_step
        self.max_in_flight = max_in_flight
        self.ready_prefetch = ready_prefetch
        self.timeout_seconds = timeout_seconds

        self._replay_lock = Lock()
        self._slots = Semaphore(max_in_flight)
        self._ready: Queue[ReadyBatch] = Queue(maxsize=ready_prefetch)
        self._runs: SimpleQueue[_Run] = SimpleQueue()
        self._controls: SimpleQueue[_PublishWeights] = SimpleQueue()
        self._control_pending = Event()
        self._stop = Event()
        self._run_done = Event()
        self._run_done.set()
        self._failure_lock = Lock()
        self._failure: BaseException | None = None
        self._outstanding_lock = Lock()
        self._outstanding: set[int] = set()
        self.max_observed_outstanding = 0
        self._next_token = 0
        self._closed = False
        self._transfer_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self._thread = Thread(
            target=self._worker_main,
            name="atariagent-batch-worker",
            daemon=True,
        )
        self._thread.start()

    @property
    def cache_size(self) -> int:
        pipeline = self.reanalysis_pipeline
        return 0 if pipeline is None else pipeline.cache_size

    @property
    def outstanding_count(self) -> int:
        with self._outstanding_lock:
            return len(self._outstanding)

    def start(self, start_step: int, count: int) -> None:
        """Start producing exactly ``count`` sequential learner batches."""
        self._require_open()
        if isinstance(start_step, bool) or not isinstance(start_step, int):
            raise TypeError("start_step must be an integer")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        if count < 0:
            raise ValueError("count must be non-negative")
        self._raise_failure()
        if not self._run_done.is_set():
            raise RuntimeError("a batch-worker run is already active")
        if self.outstanding_count:
            raise RuntimeError("previous batches have not been completed")
        if count == 0:
            return
        self._run_done.clear()
        self._runs.put(_Run(start_step=start_step, count=count))

    def next_ready(self) -> ReadyBatch:
        """Wait for the next ordered GPU batch or propagate worker failure."""
        self._require_open()
        deadline = perf_counter() + self.timeout_seconds
        while True:
            self._raise_failure()
            remaining = deadline - perf_counter()
            if remaining <= 0.0:
                raise TimeoutError("timed out waiting for a prefetched batch")
            try:
                return self._ready.get(timeout=min(0.05, remaining))
            except Empty:
                if self._run_done.is_set() and self._ready.empty():
                    self._raise_failure()

    def complete(self, ready: ReadyBatch, priorities: Tensor) -> None:
        """Apply priorities and release this batch's bounded-capacity slot."""
        self._require_open()
        with self._outstanding_lock:
            if ready.token not in self._outstanding:
                raise ValueError("batch was already completed or is not owned")
            self._outstanding.remove(ready.token)
        try:
            with self._replay_lock:
                self.replay.update_priorities(
                    ready.gpu_batch.indices, priorities
                )
        finally:
            self._slots.release()

    def publish_weights(
        self,
        version: int,
        state: Mapping[str, Tensor],
    ) -> None:
        """Order a target-weight update between old and new actor requests."""
        self._require_open()
        pipeline = self.reanalysis_pipeline
        if pipeline is None:
            return
        command = _PublishWeights(version=version, state=dict(state))
        self._controls.put(command)
        self._control_pending.set()
        deadline = perf_counter() + self.timeout_seconds
        while not command.done.wait(timeout=0.05):
            self._raise_failure()
            if perf_counter() >= deadline:
                raise TimeoutError("timed out publishing batch-worker weights")
        if command.error is not None:
            raise command.error

    def wait_idle(self) -> None:
        """Wait until the current run has produced all requested batches."""
        deadline = perf_counter() + self.timeout_seconds
        while not self._run_done.wait(timeout=0.05):
            self._raise_failure()
            if perf_counter() >= deadline:
                raise TimeoutError("timed out waiting for the batch worker")
        self._raise_failure()

    def close(self) -> None:
        """Stop the producer thread. The caller still owns the Ray pipeline."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._control_pending.set()
        self._thread.join(timeout=min(self.timeout_seconds, 5.0))

    def __enter__(self) -> BatchWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _worker_main(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_controls()
                try:
                    run = self._runs.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    self._execute_run(run)
                finally:
                    self._run_done.set()
        except _StopWorker:
            self._run_done.set()
        except BaseException as error:
            with self._failure_lock:
                self._failure = error
            self._run_done.set()
            self._fail_controls(error)

    def _execute_run(self, run: _Run) -> None:
        end_step = run.start_step + run.count
        step = run.start_step
        direct_end = (
            min(end_step, self.reanalysis_start_step)
            if self.reanalysis_pipeline is not None
            else end_step
        )
        while step < direct_end:
            self._drain_controls()
            token, batch, beta, sample_ms = self._sample(step, False)
            try:
                ready = self._transfer(
                    token,
                    step,
                    batch,
                    beta,
                    sample_duration_ms=sample_ms,
                )
                self._put_ready(ready)
            except BaseException:
                self._discard_token(token)
                raise
            step += 1

        if step < end_step:
            self._execute_reanalysis(step, end_step)

    def _execute_reanalysis(self, start_step: int, end_step: int) -> None:
        pipeline = self.reanalysis_pipeline
        if pipeline is None:
            raise RuntimeError("reanalysis pipeline is unavailable")
        submitted_step = start_step
        output_step = start_step
        pending: dict[int, tuple[int, int, float, float]] = {}
        completed: dict[int, ReadyBatch] = {}

        while output_step < end_step:
            self._drain_controls()
            while (
                submitted_step < end_step
                and pipeline.needs_prefetch
                and self.outstanding_count < self.max_in_flight
            ):
                self._drain_controls()
                token, batch, beta, sample_ms = self._sample(
                    submitted_step, True
                )
                try:
                    request_id = pipeline.submit(batch)
                except BaseException:
                    self._discard_token(token)
                    raise
                pending[request_id] = (
                    token,
                    submitted_step,
                    beta,
                    sample_ms,
                )
                submitted_step += 1

            if not pending:
                self._wait_for_capacity()
                continue

            result = pipeline.wait_next()
            token, sample_step, beta, sample_ms = pending.pop(result.request_id)
            try:
                cpu_batch = result.batch.without_value_bootstraps()
                completed[sample_step] = self._transfer(
                    token,
                    sample_step,
                    cpu_batch,
                    beta,
                    sample_duration_ms=sample_ms,
                    queue_wait_ms=result.queue_wait_ms,
                    actor_duration_ms=result.actor_duration_ms,
                    policy_roots_requested=result.policy_roots_requested,
                    policy_roots_searched=result.policy_roots_searched,
                )
            except BaseException:
                self._discard_token(token)
                raise

            while output_step in completed:
                self._put_ready(completed.pop(output_step))
                output_step += 1

    def _sample(
        self,
        sample_step: int,
        include_value_bootstraps: bool,
    ) -> tuple[int, ReplayBatch, float, float]:
        self._acquire_slot()
        started = perf_counter()
        try:
            with self._replay_lock:
                batch, beta = self.replay.sample_batch(
                    self.batch_size,
                    trained_steps=sample_step,
                    training_steps=self.training_steps,
                    include_value_bootstraps=include_value_bootstraps,
                    pin_memory=self.device.type == "cuda",
                )
            if self.device.type == "cuda" and not batch.frames.is_pinned():
                raise RuntimeError("CUDA batch-worker input must be pinned")
            token = self._next_token
            self._next_token += 1
            with self._outstanding_lock:
                self._outstanding.add(token)
                self.max_observed_outstanding = max(
                    self.max_observed_outstanding,
                    len(self._outstanding),
                )
            return token, batch, beta, (perf_counter() - started) * 1_000.0
        except BaseException:
            self._slots.release()
            raise

    def _transfer(
        self,
        token: int,
        sample_step: int,
        cpu_batch: ReplayBatch,
        beta: float,
        *,
        sample_duration_ms: float,
        queue_wait_ms: float | None = None,
        actor_duration_ms: float | None = None,
        policy_roots_requested: int = 0,
        policy_roots_searched: int = 0,
    ) -> ReadyBatch:
        started = perf_counter()
        ready_event: torch.cuda.Event | None = None
        if self._transfer_stream is None:
            gpu_batch = cpu_batch.to(
                self.device,
                keep_indices_on_cpu=True,
            )
        else:
            with torch.cuda.stream(self._transfer_stream):
                gpu_batch = cpu_batch.to(
                    self.device,
                    non_blocking=True,
                    keep_indices_on_cpu=True,
                )
                ready_event = torch.cuda.Event()
                ready_event.record(self._transfer_stream)
        return ReadyBatch(
            token=token,
            sample_step=sample_step,
            cpu_batch=cpu_batch,
            gpu_batch=gpu_batch,
            priority_beta=beta,
            ready_event=ready_event,
            sample_duration_ms=sample_duration_ms,
            transfer_enqueue_ms=(perf_counter() - started) * 1_000.0,
            queue_wait_ms=queue_wait_ms,
            actor_duration_ms=actor_duration_ms,
            policy_roots_requested=policy_roots_requested,
            policy_roots_searched=policy_roots_searched,
        )

    def _put_ready(self, ready: ReadyBatch) -> None:
        while not self._stop.is_set():
            self._drain_controls()
            try:
                self._ready.put(ready, timeout=0.05)
                return
            except Full:
                pass
        raise _StopWorker

    def _acquire_slot(self) -> None:
        while not self._stop.is_set():
            self._drain_controls()
            if self._slots.acquire(timeout=0.05):
                return
        raise _StopWorker

    def _wait_for_capacity(self) -> None:
        while self.outstanding_count >= self.max_in_flight:
            self._drain_controls()
            if self._stop.wait(0.05):
                raise _StopWorker

    def _drain_controls(self) -> None:
        if self._stop.is_set():
            raise _StopWorker
        if not self._control_pending.is_set():
            return
        while True:
            try:
                command = self._controls.get_nowait()
            except Empty:
                self._control_pending.clear()
                return
            try:
                pipeline = self.reanalysis_pipeline
                if pipeline is None:
                    raise RuntimeError("reanalysis pipeline is unavailable")
                pipeline.publish_weights(command.version, command.state)
            except BaseException as error:
                command.error = error
            finally:
                command.done.set()

    def _discard_token(self, token: int) -> None:
        with self._outstanding_lock:
            self._outstanding.discard(token)
        self._slots.release()

    def _fail_controls(self, error: BaseException) -> None:
        while True:
            try:
                command = self._controls.get_nowait()
            except Empty:
                return
            command.error = error
            command.done.set()

    def _raise_failure(self) -> None:
        with self._failure_lock:
            error = self._failure
        if error is not None:
            raise RuntimeError("batch worker failed") from error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("batch worker is closed")


__all__ = ["BatchWorker", "ReadyBatch"]
