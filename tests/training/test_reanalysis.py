from dataclasses import replace
from time import sleep

import pytest
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
    make_target_state,
)


def _batch(batch_size: int = 2) -> ReplayBatch:
    return ReplayBatch(
        frames=torch.zeros(batch_size, 5, 1, 96, 96, dtype=torch.uint8),
        actions=torch.zeros(batch_size, 1, 1, dtype=torch.long),
        rewards=torch.zeros(batch_size, 1),
        policy_targets=torch.full((batch_size, 2, 2), 0.5),
        value_targets=torch.zeros(batch_size, 2),
        action_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        policy_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        value_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size),
        value_bootstrap_frames=torch.zeros(
            batch_size, 5, 1, 96, 96, dtype=torch.uint8
        ),
        value_bootstrap_values=torch.ones(batch_size, 2),
        value_bootstrap_discounts=torch.ones(batch_size, 2),
        value_bootstrap_mask=torch.ones(batch_size, 2, dtype=torch.bool),
    )


def _pipeline(
    *,
    prefetch_batches: int = 2,
    timeout_seconds: float = 60.0,
    target_update_interval: int = 200,
) -> ReanalysisPipeline:
    pipeline = ReanalysisPipeline(
        in_channels=4,
        action_space_size=2,
        mcts_config=MCTSConfig(num_simulations=1),
        policy_chunk_size=4,
        cache_targets=True,
        rng_seed=3,
        support_min=-300,
        support_max=300,
        precision="fp32",
        mcts_threads=1,
        prefetch_batches=prefetch_batches,
        timeout_seconds=timeout_seconds,
        target_update_interval=target_update_interval,
        device="cpu",
    )
    representation = RepresentationNetwork(4)
    prediction = PredictionNetwork(2)
    dynamics = DynamicsNetwork(2)
    pipeline.publish_weights(
        0,
        make_target_state(representation, prediction, dynamics),
        wait=True,
    )
    return pipeline


def test_native_pipeline_enforces_prefetch_bound_and_matches_requests() -> None:
    pipeline = _pipeline(prefetch_batches=2)
    try:
        first = pipeline.submit(_batch())
        second = pipeline.submit(replace(_batch(), indices=torch.tensor([2, 3])))
        assert (first, second) == (0, 1)
        assert pipeline.pending_count == 2
        assert pipeline.max_observed_pending == 2
        assert pipeline.max_observed_pending_bytes > 0
        with pytest.raises(BufferError, match="prefetch limit"):
            pipeline.submit(_batch())

        ready_ids = {pipeline.wait_next().request_id, pipeline.wait_next().request_id}
        assert ready_ids == {0, 1}
        assert pipeline.pending_count == 0
    finally:
        pipeline.close()


def test_native_pipeline_reuses_cache_and_clears_on_weights() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    batch = _batch()
    try:
        pipeline.submit(batch)
        first = pipeline.wait_next()
        pipeline.submit(batch)
        cached = pipeline.wait_next()

        assert first.policy_roots_requested == 4
        assert first.policy_roots_searched == 3
        assert cached.policy_roots_searched == 0
        assert pipeline.cache_size == 3
        torch.testing.assert_close(
            cached.batch.value_targets,
            first.batch.value_targets,
        )
        torch.testing.assert_close(
            cached.batch.policy_targets,
            first.batch.policy_targets,
        )

        representation = RepresentationNetwork(4)
        prediction = PredictionNetwork(2)
        dynamics = DynamicsNetwork(2)
        pipeline.publish_weights(
            1,
            make_target_state(representation, prediction, dynamics),
        )
        assert pipeline.cache_size == 0
        assert pipeline.weight_version == 1
    finally:
        pipeline.close()


def test_native_pipeline_validates_native_batch_tensor_contract() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    try:
        with pytest.raises(ValueError, match="policy_mask must be a 2D bool"):
            pipeline.submit(
                replace(_batch(), policy_mask=torch.ones(2, 2))
            )
        with pytest.raises(ValueError, match="indices must be a 1D int64"):
            pipeline.submit(
                replace(_batch(), indices=torch.arange(2, dtype=torch.int32))
            )
    finally:
        pipeline.close()


def test_native_pipeline_retains_tensor_storage_without_transport_copy() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    batch = _batch()
    frame_pointer = batch.frames.data_ptr()
    try:
        pipeline.submit(batch)
        ready = pipeline.wait_next()

        assert ready.batch.frames.data_ptr() == frame_pointer
        assert ready.transfer_duration_ms == 0.0
        assert ready.worker_duration_ms >= 0.0
        assert ready.queue_wait_ms >= 0.0
    finally:
        pipeline.close()


def test_native_pipeline_validates_ordering_timeout_and_shutdown() -> None:
    with pytest.raises(ValueError, match="target_update_interval"):
        _pipeline(target_update_interval=0)

    with pytest.raises(ValueError, match="prefetch_batches"):
        ReanalysisPipeline(
            in_channels=4,
            action_space_size=2,
            mcts_config=MCTSConfig(num_simulations=1),
            policy_chunk_size=4,
            cache_targets=True,
            rng_seed=0,
            support_min=-300,
            support_max=300,
            precision="fp32",
            mcts_threads=1,
            prefetch_batches=0,
            timeout_seconds=1.0,
            target_update_interval=200,
            device="cpu",
        )

    pipeline = _pipeline(prefetch_batches=1, timeout_seconds=0.01)
    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.submit(_batch())
    pipeline.close()
