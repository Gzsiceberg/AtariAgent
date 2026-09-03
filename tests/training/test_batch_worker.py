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
        self.priority_betas: list[float] = []

    def sample(
        self,
        batch_size: int,
        *,
        include_value_bootstraps: bool,
        pin_memory: bool,
        priority_beta: float,
    ):
        assert batch_size == 2
        assert not pin_memory
        self.priority_betas.append(priority_beta)
        return self._sample(include_value_bootstraps)

    def update_priorities(
        self, indices: torch.Tensor, priorities: torch.Tensor
    ) -> None:
        self.priority_updates.append((indices.clone(), priorities.clone()))


@pytest.mark.parametrize("interval", [True, 2.0])
def test_worker_rejects_non_integer_cache_clear_interval(interval: object) -> None:
    with pytest.raises(
        TypeError,
        match="reanalysis_initial_cache_clear_interval",
    ):
        BatchWorker(
            _FakeReplay(lambda _: _batch()),  # type: ignore[arg-type]
            batch_size=2,
            device="cpu",
            reanalysis_initial_cache_clear_interval=interval,  # type: ignore[arg-type]
        )


def test_worker_rejects_non_positive_cache_clear_interval() -> None:
    with pytest.raises(ValueError, match="cache clear intervals"):
        BatchWorker(
            _FakeReplay(lambda _: _batch()),  # type: ignore[arg-type]
            batch_size=2,
            device="cpu",
            reanalysis_initial_cache_clear_interval=0,
        )


def test_worker_bounds_sampling_when_consumer_is_slow() -> None:
    sampled: list[int] = []
    lock = Lock()

    def sample(include: bool):
        assert not include
        with lock:
            step = len(sampled) + 10
            sampled.append(step)
        return _batch(step).without_value_bootstraps()

    replay = _FakeReplay(sample)
    with BatchWorker(
        replay,  # type: ignore[arg-type]
        batch_size=2,
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


def test_worker_anneals_priority_beta_by_learner_step() -> None:
    replay = _FakeReplay(lambda _: _batch().without_value_bootstraps())
    with BatchWorker(
        replay,  # type: ignore[arg-type]
        batch_size=2,
        device="cpu",
        priority_beta_initial=0.4,
        priority_beta_final=1.0,
        priority_beta_steps=100,
        max_in_flight=1,
        ready_prefetch=1,
        timeout_seconds=2.0,
    ) as worker:
        worker.start(50, 1)
        ready = worker.next_ready()
        worker.complete(ready, torch.ones(2))
        worker.wait_idle()

    assert replay.priority_betas == pytest.approx([0.7])


def test_worker_propagates_sampling_failure() -> None:
    first_sampled = Event()

    sampled = 0

    def sample(include: bool):
        nonlocal sampled
        del include
        step = sampled
        sampled += 1
        if step == 1:
            raise ValueError("sample failed")
        first_sampled.set()
        return _batch(step).without_value_bootstraps()

    worker = BatchWorker(
        _FakeReplay(sample),  # type: ignore[arg-type]
        batch_size=2,
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
    def __init__(self, mcts_bootstrap_start_step: int | None = None) -> None:
        self.prefetch_batches = 2
        self.mcts_bootstrap_start_step = mcts_bootstrap_start_step
        self.pending: list[tuple[int, ReplayBatch]] = []
        self.next_request_id = 0
        self.published_versions: list[int] = []
        self.cache_clear_request_counts: list[int] = []
        self.submitted_steps: list[int] = []
        self.cache_size = 0

    @property
    def needs_prefetch(self) -> bool:
        return len(self.pending) < self.prefetch_batches

    def uses_mcts_bootstrap(self, trained_steps: int) -> bool:
        return (
            self.mcts_bootstrap_start_step is not None
            and trained_steps >= self.mcts_bootstrap_start_step
        )

    def submit(self, batch: ReplayBatch, *, trained_steps: int = 0) -> int:
        self.submitted_steps.append(trained_steps)
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

    def clear_cache(self) -> None:
        self.cache_clear_request_counts.append(self.next_request_id)


def test_worker_applies_mixed_values_before_learner_transfer() -> None:
    pipeline = _FakeReanalysisPipeline()

    def sample(include: bool):
        assert include
        batch = _batch(30)
        batch = replace(
            batch,
            search_value_targets=torch.tensor(
                [[10.0, 20.0], [30.0, 40.0]]
            ),
            transition_ages=torch.tensor([5_000, 4_999]),
        )
        return batch

    with BatchWorker(
        _FakeReplay(sample),  # type: ignore[arg-type]
        batch_size=2,
        device="cpu",
        reanalysis_pipeline=pipeline,  # type: ignore[arg-type]
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


def test_worker_mcts_bootstrap_overrides_mixed_targets_with_td() -> None:
    pipeline = _FakeReanalysisPipeline(mcts_bootstrap_start_step=100)
    submitted_masks: list[torch.Tensor] = []
    original_submit = pipeline.submit

    def submit(batch: ReplayBatch, *, trained_steps: int = 0) -> int:
        assert batch.mcts_bootstrap_mask is not None
        submitted_masks.append(batch.mcts_bootstrap_mask.clone())
        return original_submit(batch, trained_steps=trained_steps)

    pipeline.submit = submit  # type: ignore[method-assign]

    def sample(include: bool):
        assert include
        return replace(
            _batch(),
            search_value_targets=torch.ones(2, 2),
            transition_ages=torch.tensor([0, 5_000]),
        )

    with BatchWorker(
        _FakeReplay(sample),  # type: ignore[arg-type]
        batch_size=2,
        device="cpu",
        reanalysis_pipeline=pipeline,  # type: ignore[arg-type]
        value_target="mixed",
        collection_steps=100,
        mixed_value_start_step=30,
        mixed_value_threshold=5_000,
        max_in_flight=1,
        ready_prefetch=1,
        timeout_seconds=2.0,
    ) as worker:
        worker.start(100, 1)
        ready = worker.next_ready()
        torch.testing.assert_close(
            ready.gpu_batch.value_targets,
            torch.zeros(2, 2),
        )
        worker.complete(ready, torch.ones(2))
        worker.wait_idle()

    torch.testing.assert_close(
        submitted_masks[0],
        torch.tensor([[False, False], [True, True]]),
    )


def test_worker_reanalyzes_in_order_and_ramps_cache_clearing() -> None:
    includes: list[bool] = []
    pipeline = _FakeReanalysisPipeline()

    def sample(include: bool):
        assert include
        step = len(includes)
        includes.append(include)
        return _batch(step)

    with BatchWorker(
        _FakeReplay(sample),  # type: ignore[arg-type]
        batch_size=2,
        device="cpu",
        reanalysis_pipeline=pipeline,  # type: ignore[arg-type]
        reanalysis_initial_cache_clear_interval=2,
        reanalysis_final_cache_clear_interval=4,
        reanalysis_cache_clear_ramp_steps=4,
        max_in_flight=3,
        ready_prefetch=1,
        timeout_seconds=2.0,
    ) as worker:
        worker.start(0, 10)
        steps = []
        for _ in range(10):
            ready = worker.next_ready()
            steps.append(ready.sample_step)
            assert ready.worker_duration_ms == pytest.approx(2.0)
            assert ready.gpu_batch.value_bootstrap_frames is None
            worker.complete(ready, torch.ones(2))
        worker.wait_idle()
        worker.publish_weights(10, {})

    assert steps == list(range(10))
    assert includes == [True] * 10
    assert pipeline.submitted_steps == list(range(10))
    assert pipeline.cache_clear_request_counts == [2, 5, 9]
    assert pipeline.published_versions == [10]
