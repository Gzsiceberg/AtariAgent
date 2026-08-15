from dataclasses import replace
from threading import Event, Lock
from time import monotonic, sleep

import pytest
import torch

from atariagent.replay_batch import ReplayBatch
from atariagent.training import BatchWorker
from atariagent.training.reanalysis import ReadyReanalysis


def _batch(index: int = 0) -> ReplayBatch:
    return ReplayBatch(
        frames=torch.full((2, 2, 1, 1, 1), index, dtype=torch.uint8),
        actions=torch.zeros(2, 1, 1, dtype=torch.long),
        rewards=torch.zeros(2, 1),
        policy_targets=torch.full((2, 2, 2), 0.5),
        value_targets=torch.zeros(2, 2),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        policy_mask=torch.ones(2, 2, dtype=torch.bool),
        value_mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.tensor([index * 2, index * 2 + 1]),
        importance_weights=torch.ones(2),
        value_bootstrap_frames=torch.zeros(2, 2, 1, 1, 1, dtype=torch.uint8),
        value_bootstrap_values=torch.zeros(2, 2),
        value_bootstrap_discounts=torch.ones(2, 2),
        value_bootstrap_mask=torch.ones(2, 2, dtype=torch.bool),
    )


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while not predicate():
        if monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        sleep(0.005)


class _FakeReplay:
    def __init__(self, sample) -> None:
        self._sample = sample
        self.priority_updates: list[tuple[torch.Tensor, torch.Tensor]] = []

    def sample_batch(
        self,
        batch_size: int,
        *,
        trained_steps: int,
        training_steps: int,
        include_value_bootstraps: bool,
        pin_memory: bool,
    ):
        assert batch_size == 2
        assert training_steps == 100
        assert not pin_memory
        return self._sample(trained_steps, include_value_bootstraps)

    def update_priorities(
        self, indices: torch.Tensor, priorities: torch.Tensor
    ) -> None:
        self.priority_updates.append((indices.clone(), priorities.clone()))


def test_worker_bounds_sampling_when_consumer_is_slow() -> None:
    sampled: list[int] = []
    lock = Lock()

    def sample(step: int, include: bool):
        assert not include
        with lock:
            sampled.append(step)
        return _batch(step).without_value_bootstraps(), 0.4

    replay = _FakeReplay(sample)
    with BatchWorker(
        replay,  # type: ignore[arg-type]
        batch_size=2,
        training_steps=100,
        device="cpu",
        max_in_flight=2,
        ready_prefetch=1,
        timeout_seconds=2.0,
    ) as worker:
        worker.start(10, 5)
        _wait_until(lambda: len(sampled) == 2)
        sleep(0.05)
        assert sampled == [10, 11]
        assert worker.outstanding_count == 2

        ready_steps: list[int] = []
        for _ in range(5):
            ready = worker.next_ready()
            ready_steps.append(ready.sample_step)
            worker.complete(ready, torch.ones(2))
        with pytest.raises(ValueError, match="already completed"):
            worker.complete(ready, torch.ones(2))
        worker.wait_idle()

    assert ready_steps == [10, 11, 12, 13, 14]
    assert sampled == ready_steps
    assert len(replay.priority_updates) == 5


def test_worker_propagates_sampling_failure() -> None:
    first_sampled = Event()

    def sample(step: int, include: bool):
        del include
        if step == 1:
            raise ValueError("sample failed")
        first_sampled.set()
        return _batch(step).without_value_bootstraps(), 0.4

    worker = BatchWorker(
        _FakeReplay(sample),  # type: ignore[arg-type]
        batch_size=2,
        training_steps=100,
        device="cpu",
        max_in_flight=2,
        ready_prefetch=1,
        timeout_seconds=2.0,
    )
    try:
        worker.start(0, 2)
        assert first_sampled.wait(1.0)
        ready = worker.next_ready()
        worker.complete(ready, torch.ones(2))
        with pytest.raises(RuntimeError, match="batch worker failed") as error:
            worker.next_ready()
        assert isinstance(error.value.__cause__, ValueError)
    finally:
        worker.close()


class _FakeReanalysisPipeline:
    def __init__(self) -> None:
        self.prefetch_batches = 2
        self.pending: list[tuple[int, ReplayBatch]] = []
        self.next_request_id = 0
        self.published_versions: list[int] = []
        self.cache_size = 0

    @property
    def needs_prefetch(self) -> bool:
        return len(self.pending) < self.prefetch_batches

    def submit(self, batch: ReplayBatch) -> int:
        request_id = self.next_request_id
        self.next_request_id += 1
        self.pending.append((request_id, batch))
        return request_id

    def wait_next(self) -> ReadyReanalysis:
        request_id, batch = self.pending.pop(0)
        return ReadyReanalysis(
            request_id=request_id,
            batch=batch,
            weight_version=0,
            queue_wait_ms=1.0,
            worker_duration_ms=2.0,
            transfer_duration_ms=0.0,
            peak_memory_bytes=0,
            policy_roots_requested=4,
            policy_roots_searched=3,
        )

    def publish_weights(self, version: int, state) -> None:
        del state
        self.published_versions.append(version)


def test_worker_applies_mixed_values_before_learner_transfer() -> None:
    pipeline = _FakeReanalysisPipeline()

    def sample(step: int, include: bool):
        assert include
        batch = _batch(step)
        batch = replace(
            batch,
            search_value_targets=torch.tensor(
                [[10.0, 20.0], [30.0, 40.0]]
            ),
            transition_ages=torch.tensor([5_000, 4_999]),
        )
        return batch, 0.5

    with BatchWorker(
        _FakeReplay(sample),  # type: ignore[arg-type]
        batch_size=2,
        training_steps=100,
        device="cpu",
        reanalysis_pipeline=pipeline,  # type: ignore[arg-type]
        reanalysis_start_step=0,
        value_target="mixed",
        mixed_value_start_step=30,
        mixed_value_threshold=5_000,
        max_in_flight=1,
        ready_prefetch=1,
        timeout_seconds=2.0,
    ) as worker:
        worker.start(30, 1)
        ready = worker.next_ready()
        torch.testing.assert_close(
            ready.gpu_batch.value_targets,
            torch.tensor([[10.0, 20.0], [0.0, 0.0]]),
        )
        assert ready.gpu_batch.search_value_targets is None
        assert ready.gpu_batch.transition_ages is None
        worker.complete(ready, torch.ones(2))
        worker.wait_idle()


def test_worker_handles_direct_then_reanalysis_batches_in_order() -> None:
    includes: list[bool] = []
    pipeline = _FakeReanalysisPipeline()

    def sample(step: int, include: bool):
        includes.append(include)
        batch = _batch(step)
        return (batch if include else batch.without_value_bootstraps()), 0.5

    with BatchWorker(
        _FakeReplay(sample),  # type: ignore[arg-type]
        batch_size=2,
        training_steps=100,
        device="cpu",
        reanalysis_pipeline=pipeline,  # type: ignore[arg-type]
        reanalysis_start_step=2,
        max_in_flight=3,
        ready_prefetch=1,
        timeout_seconds=2.0,
    ) as worker:
        worker.start(0, 5)
        steps = []
        for _ in range(5):
            ready = worker.next_ready()
            steps.append(ready.sample_step)
            if ready.sample_step >= 2:
                assert ready.worker_duration_ms == pytest.approx(2.0)
                assert ready.gpu_batch.value_bootstrap_frames is None
            worker.complete(ready, torch.ones(2))
        worker.wait_idle()
        worker.publish_weights(10, {})

    assert steps == [0, 1, 2, 3, 4]
    assert includes == [False, False, True, True, True]
    assert pipeline.published_versions == [10]
