from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest
import ray
import torch

from atariagent.models import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig
from atariagent.training.reanalysis import (
    ReanalysisPipeline,
    ReanalysisResult,
    create_reanalysis_actor,
    initialize_local_ray,
    make_target_state,
)


@dataclass(frozen=True)
class _FakeRef:
    identifier: int


class _FakeRay:
    def __init__(self) -> None:
        self.objects: dict[_FakeRef, Any] = {}
        self.next_identifier = 0
        self.killed = False
        self.killed_count = 0
        self.time_out = False
        self.actors: list[Any] = []

    def put(self, value: Any) -> _FakeRef:
        reference = _FakeRef(self.next_identifier)
        self.next_identifier += 1
        self.objects[reference] = value
        return reference

    def get(self, reference: _FakeRef) -> Any:
        return self.objects[reference]

    def wait(self, references, *, num_returns, timeout):
        del timeout
        if self.time_out:
            return [], list(references)
        return list(references)[:num_returns], list(references)[num_returns:]

    def kill(self, actor, *, no_restart):
        del actor, no_restart
        self.killed = True
        self.killed_count += 1


class _RemoteMethod:
    def __init__(self, callback) -> None:
        self.callback = callback

    def remote(self, *args):
        return self.callback(*args)


class _FakeActor:
    def __init__(self, ray_api: _FakeRay) -> None:
        self.ray = ray_api
        self.version = -1
        self.request_ids: list[int] = []
        self.requests: list[Any] = []
        self.cache: dict[int, tuple[np.float32, np.ndarray]] = {}
        self.set_weights = _RemoteMethod(self._set_weights)
        self.reanalyze = _RemoteMethod(self._reanalyze)

    def _set_weights(self, version, state):
        del state
        self.version = version
        self.cache.clear()
        return self.ray.put(version)

    def _reanalyze(self, request):
        self.request_ids.append(request.request_id)
        self.requests.append(request)
        value_targets = request.value_targets.copy()
        policy_targets = request.policy_targets.copy()
        requested = int(request.policy_mask.sum())
        searched = 0
        positions_by_id: dict[int, list[tuple[int, int]]] = {}
        for sample, offset in np.argwhere(request.policy_mask):
            state_id = int(request.indices[sample]) + int(offset)
            positions_by_id.setdefault(state_id, []).append(
                (int(sample), int(offset))
            )
        for state_id, positions in positions_by_id.items():
            cached = self.cache.get(state_id)
            if cached is None:
                source = positions[0]
                value = np.float32(value_targets[source] + 1.0)
                policy = np.asarray([1.0, 0.0], dtype=np.float32)
                self.cache[state_id] = (value, policy)
                searched += 1
            else:
                value, policy = cached
            for position in positions:
                value_targets[position] = value
                policy_targets[position] = policy
        return self.ray.put(
            ReanalysisResult(
                request_id=request.request_id,
                weight_version=request.weight_version,
                value_targets=value_targets,
                policy_targets=policy_targets,
                actor_duration_ms=2.0,
                transfer_duration_ms=0.5,
                peak_memory_bytes=0,
                policy_roots_requested=requested,
                policy_roots_searched=searched,
                cache_size=len(self.cache),
            )
        )


def _batch(batch_size: int = 2) -> ReplayBatch:
    return ReplayBatch(
        frames=torch.zeros(batch_size, 2, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(batch_size, 1, 1, dtype=torch.long),
        rewards=torch.zeros(batch_size, 1),
        policy_targets=torch.full((batch_size, 2, 2), 0.5),
        value_targets=torch.zeros(batch_size, 2),
        action_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        policy_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        value_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size),
    )


def _fake_pipeline(*, max_weight_lag: int = 200):
    ray_api = _FakeRay()
    actor = _FakeActor(ray_api)
    ray_api.actors = [actor]
    pipeline = ReanalysisPipeline(
        actor,
        prefetch_batches=2,
        timeout_seconds=1.0,
        max_weight_lag=max_weight_lag,
        ray_api=ray_api,
    )
    pipeline.publish_weights(0, {}, wait=True)
    return pipeline, ray_api


def test_pipeline_enforces_prefetch_limit_for_individual_requests() -> None:
    pipeline, ray_api = _fake_pipeline()

    request_ids = [pipeline.submit(_batch()) for _ in range(2)]
    assert request_ids == [0, 1]
    assert pipeline.pending_count == 2
    assert pipeline.max_observed_pending == 2
    assert pipeline.max_observed_pending_bytes > 0
    assert pipeline.pending_payload_bytes == pipeline.max_observed_pending_bytes
    assert not pipeline.needs_prefetch
    with pytest.raises(BufferError, match="prefetch limit"):
        pipeline.submit(_batch())

    ready = pipeline.wait_next()

    assert ready.request_id == 0
    assert ready.actor_duration_ms == pytest.approx(2.0)
    torch.testing.assert_close(ready.batch.value_targets, torch.ones(2, 2))
    assert pipeline.pending_count == 1
    assert pipeline.cache_size == 3
    assert pipeline.needs_prefetch
    assert ray_api.actors[0].request_ids == [0, 1]
    pipeline.close()
    assert ray_api.killed
    assert ray_api.killed_count == 1


def test_worker_cache_reuses_targets_across_queued_batches() -> None:
    pipeline, _ = _fake_pipeline()
    pipeline.submit(_batch())
    pipeline.submit(_batch())

    first_ready = pipeline.wait_next()
    second_ready = pipeline.wait_next()

    assert first_ready.policy_roots_requested == 4
    assert first_ready.policy_roots_searched == 3
    assert second_ready.policy_roots_requested == 4
    assert second_ready.policy_roots_searched == 0
    torch.testing.assert_close(
        second_ready.batch.value_targets,
        first_ready.batch.value_targets,
    )
    torch.testing.assert_close(
        second_ready.batch.policy_targets,
        first_ready.batch.policy_targets,
    )
    pipeline.close()


def test_pipeline_caches_value_and_policy_together() -> None:
    pipeline, ray_api = _fake_pipeline()
    batch = replace(
        _batch(),
        value_bootstrap_frames=torch.zeros(2, 2, 1, 1, 1, dtype=torch.uint8),
        value_bootstrap_values=torch.zeros(2, 2),
        value_bootstrap_discounts=torch.ones(2, 2),
        value_bootstrap_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    pipeline.submit(batch)

    ready = pipeline.wait_next()

    assert pipeline.cache_size == 3
    assert ready.batch.value_targets[1, 0] == ready.batch.value_targets[0, 1]
    pipeline.submit(batch)
    cached = pipeline.wait_next()
    assert cached.policy_roots_requested == 4
    assert cached.policy_roots_searched == 0
    torch.testing.assert_close(cached.batch.value_targets, ready.batch.value_targets)
    torch.testing.assert_close(
        cached.batch.policy_targets, ready.batch.policy_targets
    )
    pipeline.close()


def test_worker_clears_target_cache_for_new_target_weights() -> None:
    pipeline, _ = _fake_pipeline()
    pipeline.submit(_batch())
    pipeline.wait_next()
    assert pipeline.cache_size == 3

    pipeline.publish_weights(1, {})

    assert pipeline.cache_size == 0
    pipeline.close()


def test_pipeline_rejects_stale_results_and_times_out() -> None:
    pipeline, ray_api = _fake_pipeline(max_weight_lag=10)
    pipeline.submit(_batch())
    pipeline.publish_weights(20, {})
    with pytest.raises(RuntimeError, match="lag 20"):
        pipeline.wait_next()
    assert pipeline.cache_size == 0
    pipeline.close()

    pipeline, ray_api = _fake_pipeline()
    pipeline.submit(_batch())
    ray_api.time_out = True
    with pytest.raises(TimeoutError, match="timed out"):
        pipeline.wait_next()
    pipeline.close()


def test_local_ray_actor_reanalyzes_value_targets() -> None:
    owns_ray = initialize_local_ray()
    actor = None
    pipeline = None
    try:
        representation = RepresentationNetwork(4)
        prediction = PredictionNetwork(action_space_size=2)
        dynamics = DynamicsNetwork(action_space_size=2)
        actor = create_reanalysis_actor(
            num_gpus=0.0,
            in_channels=4,
            action_space_size=2,
            mcts_config=MCTSConfig(num_simulations=1),
            policy_chunk_size=2,
            cache_targets=True,
            rng_seed=3,
            support_min=-300,
            support_max=300,
            precision="fp32",
        )
        pipeline = ReanalysisPipeline(
            actor,
            prefetch_batches=1,
            timeout_seconds=60.0,
            max_weight_lag=0,
        )
        pipeline.publish_weights(
            0,
            make_target_state(representation, prediction, dynamics),
            wait=True,
        )
        batch = replace(
            _batch(),
            frames=torch.zeros(2, 5, 1, 96, 96, dtype=torch.uint8),
            value_targets=torch.ones(2, 2),
            value_bootstrap_frames=torch.zeros(
                2, 5, 1, 96, 96, dtype=torch.uint8
            ),
            value_bootstrap_values=torch.ones(2, 2),
            value_bootstrap_discounts=torch.ones(2, 2),
            value_bootstrap_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        pipeline.submit(batch)

        ready = pipeline.wait_next()
        pipeline.submit(batch)
        cached = pipeline.wait_next()

        torch.testing.assert_close(
            ready.batch.value_targets,
            torch.zeros(2, 2),
            atol=1e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            cached.batch.value_targets,
            ready.batch.value_targets,
        )
        assert ready.policy_roots_searched == 3
        assert cached.policy_roots_searched == 0
        assert ready.weight_version == 0
    finally:
        if pipeline is not None:
            pipeline.close()
        elif actor is not None:
            ray.kill(actor, no_restart=True)
        if owns_ray and ray.is_initialized():
            ray.shutdown()
