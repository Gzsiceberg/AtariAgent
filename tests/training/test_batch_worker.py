from dataclasses import replace
from threading import Event, Lock
from time import monotonic, sleep

import numpy as np
import pytest
import torch

from atariagent.replay import FIFOReplayBuffer
from atariagent.replay_batch import ReplayBatch
from atariagent.search import SearchResult
from atariagent.selfplay import GameTrajectory
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
        self.priorities = np.array([9.0, 0.125, 3.0])

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


@pytest.mark.parametrize("valid_actions", [[True, False], [False, True], [False, False], [True, True]])
def test_worker_updates_every_root_independent_of_recurrent_mask(valid_actions) -> None:
    batch = replace(
        _batch().without_value_bootstraps(),
        action_mask=torch.tensor(valid_actions)[:, None],
    )
    replay = _FakeReplay(lambda _: batch)
    candidates = torch.tensor([34.586, 123.0], requires_grad=True)
    original = candidates.clone()
    with BatchWorker(
        replay,  # type: ignore[arg-type]
        batch_size=2,
        device="cpu",
        max_in_flight=1,
        ready_prefetch=1,
        timeout_seconds=2.0,
    ) as worker:
        worker.start(0, 1)
        ready = worker.next_ready()
        worker.complete(ready, candidates)
        worker.wait_idle()
        assert worker.outstanding_count == 0

    assert len(replay.priority_updates) == 1
    indices, priorities = replay.priority_updates[0]
    torch.testing.assert_close(indices, batch.indices)
    torch.testing.assert_close(priorities, candidates)
    torch.testing.assert_close(candidates, original)
    assert not priorities.requires_grad


@pytest.mark.parametrize("terminated", [False, True])
def test_worker_updates_priorities_for_timeout_and_terminal_tails(
    terminated: bool,
) -> None:
    replay = FIFOReplayBuffer(4, unroll_steps=2)
    replay.add(GameTrajectory(
        environment_index=0, episode_id=0, block_id=0, stack_size=1,
        frames=tuple(np.zeros((1, 2, 2), dtype=np.uint8) for _ in range(5)),
        actions=(0,) * 4, rewards=(0.0,) * 4, raw_rewards=(0.0,) * 4,
        search_results=tuple(
            SearchResult(action=0, policy_target=(0.5, 0.5), root_value=0.0)
            for _ in range(4)
        ),
        predicted_values=(1.0,) * 4,
        terminated=terminated, truncated=not terminated, full_episode_done=True,
    ))
    with BatchWorker(
        replay, batch_size=4, device="cpu", max_in_flight=1,
        ready_prefetch=1, timeout_seconds=2.0,
    ) as worker:
        worker.start(0, 1)
        ready = worker.next_ready()
        # Missing bootstraps no longer make timeout-tail roots invalid.
        assert torch.all(ready.cpu_batch.value_targets[:, 0] == 0)
        assert ready.cpu_batch.value_mask[:, 0].all()
        ids = ready.cpu_batch.indices
        candidates = (ids + 1).float() / 10
        worker.complete(ready, candidates)
        worker.wait_idle()

    expected = [0.1, 0.2, 0.3, 0.4]
    np.testing.assert_allclose(replay.priorities, expected)


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
    def __init__(self) -> None:
        self.prefetch_batches = 2
        self.pending: list[tuple[int, ReplayBatch]] = []
        self.next_request_id = 0
        self.published_versions: list[int] = []
        self.cache_clear_request_counts: list[int] = []
        self.submitted_steps: list[int] = []
        self.cache_size = 0

    @property
    def needs_prefetch(self) -> bool:
        return len(self.pending) < self.prefetch_batches

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
            transition_ages=torch.tensor([4_999, 4_998]),
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


def test_worker_reanalyzes_in_order_without_scheduled_cache_clearing() -> None:
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
        max_in_flight=3,
        ready_prefetch=1,
        timeout_seconds=2.0,
    ) as worker:
        worker.start(990, 10)
        steps = []
        for _ in range(10):
            ready = worker.next_ready()
            steps.append(ready.sample_step)
            assert ready.worker_duration_ms == pytest.approx(2.0)
            assert ready.gpu_batch.value_bootstrap_frames is None
            worker.complete(ready, torch.ones(2))
        worker.wait_idle()
        worker.publish_weights(1_000, {})

    assert steps == list(range(990, 1_000))
    assert includes == [True] * 10
    assert pipeline.submitted_steps == list(range(990, 1_000))
    assert pipeline.cache_clear_request_counts == []
    assert pipeline.published_versions == [1_000]
