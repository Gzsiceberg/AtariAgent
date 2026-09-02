from dataclasses import replace

import pytest
import torch

from atariagent.models import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.search import SearchConfig
from atariagent.training.reanalysis import (
    ReanalysisPipeline,
    make_target_state,
    replay_batch_nbytes,
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
        reanalysis_state_ids=(
            torch.arange(batch_size)[:, None] + torch.arange(2)[None, :]
        ),
    )


def _pipeline(
    *,
    prefetch_batches: int = 2,
    timeout_seconds: float = 60.0,
    target_update_interval: int = 200,
    search_algorithm: str = "puct",
    cache_target_ttl: int = 200,
) -> ReanalysisPipeline:
    pipeline = ReanalysisPipeline(
        in_channels=4,
        action_space_size=2,
        search_config=SearchConfig(
            num_simulations=2 if search_algorithm == "gumbel" else 1,
            search_algorithm=search_algorithm,
            num_top_actions=2,
        ),
        policy_chunk_size=4,
        cache_targets=True,
        cache_target_ttl=cache_target_ttl,
        rng_seed=3,
        support_min=-300,
        support_max=300,
        precision="fp32",
        search_threads=1,
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


@pytest.mark.parametrize("search_algorithm", ["puct", "gumbel"])
def test_native_pipeline_configures_search_noise_from_algorithm(
    search_algorithm: str,
) -> None:
    pipeline = _pipeline(
        prefetch_batches=1,
        search_algorithm=search_algorithm,
    )
    try:
        expected_temperature = 0.0 if search_algorithm == "gumbel" else 1.0
        assert pipeline.root_noise_temperature() == expected_temperature
        pipeline.submit(_batch(), trained_steps=50)
        assert pipeline.wait_next().policy_roots_searched == 3
    finally:
        pipeline.close()


def test_policy_reanalysis_replaces_stored_targets_immediately() -> None:
    pipeline = _pipeline(prefetch_batches=1, cache_target_ttl=0)
    batch = _batch()
    batch = replace(
        batch,
        policy_targets=torch.zeros_like(batch.policy_targets),
    )
    try:
        pipeline.submit(batch, trained_steps=0)
        reanalyzed = pipeline.wait_next().batch

        assert not torch.equal(reanalyzed.policy_targets, batch.policy_targets)
    finally:
        pipeline.close()


def test_native_pipeline_accepts_consolidated_reanalysis_frames() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    batch = _batch()
    combined = torch.zeros(batch.batch_size, 6, 1, 96, 96, dtype=torch.uint8)
    shared_batch = replace(
        batch,
        frames=combined[:, :5],
        value_bootstrap_frames=combined[:, 1:],
        reanalysis_frames=combined,
    )
    assert replay_batch_nbytes(shared_batch) < replay_batch_nbytes(batch)
    try:
        pipeline.submit(shared_batch)
        ready = pipeline.wait_next()
        assert ready.batch.search_value_targets is not None
        assert ready.policy_roots_searched == 3
    finally:
        pipeline.close()


def test_native_pipeline_uses_explicit_state_ids_instead_of_id_arithmetic() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    batch = replace(
        _batch(),
        reanalysis_state_ids=torch.tensor([[0, 10], [11, 12]]),
    )
    try:
        pipeline.submit(batch)
        first = pipeline.wait_next()
        pipeline.submit(batch)
        cached = pipeline.wait_next()

        # The old index-plus-offset keys aliased (sample 0, offset 1) with
        # (sample 1, offset 0). Their explicit logical state IDs are distinct.
        assert first.policy_roots_searched == 4
        assert first.cache_hits == 0
        assert cached.policy_roots_searched == 0
        assert cached.cache_hits == 4
        assert pipeline.cache_size == 4
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
        assert cached.cache_hits == 4
        assert cached.cache_target_age_mean == pytest.approx(0.0)
        assert cached.cache_target_age_max == 0
        assert pipeline.cache_size == 3
        torch.testing.assert_close(
            cached.batch.value_targets,
            first.batch.value_targets,
        )
        torch.testing.assert_close(
            cached.batch.policy_targets,
            first.batch.policy_targets,
        )
        assert cached.batch.search_value_targets is not None
        assert first.batch.search_value_targets is not None
        torch.testing.assert_close(
            cached.batch.search_value_targets,
            first.batch.search_value_targets,
        )

        pipeline.clear_cache()
        pipeline.submit(batch)
        refreshed = pipeline.wait_next()
        assert refreshed.policy_roots_searched == 3
        assert pipeline.cache_size == 3

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


def test_native_pipeline_expires_cached_targets_at_ttl() -> None:
    pipeline = _pipeline(
        prefetch_batches=1,
        cache_target_ttl=200,
    )
    batch = _batch()
    try:
        pipeline.submit(batch, trained_steps=0)
        pipeline.wait_next()
        pipeline.submit(batch, trained_steps=199)
        cached = pipeline.wait_next()
        pipeline.submit(batch, trained_steps=200)
        expired = pipeline.wait_next()

        assert cached.policy_roots_searched == 0
        assert cached.cache_hits == 4
        assert cached.cache_target_age_mean == pytest.approx(199.0)
        assert cached.cache_target_age_max == 199
        assert expired.policy_roots_searched == 3
        assert expired.cache_hits == 0
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
        with pytest.raises(ValueError, match="reanalysis_state_ids"):
            pipeline.submit(replace(_batch(), reanalysis_state_ids=None))
        with pytest.raises(ValueError, match="reanalysis_state_ids"):
            pipeline.submit(
                replace(
                    _batch(),
                    reanalysis_state_ids=torch.zeros(2, 2, dtype=torch.int32),
                )
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
    with pytest.raises(ValueError, match="cache_target_ttl"):
        _pipeline(cache_target_ttl=-1)

    with pytest.raises(ValueError, match="prefetch_batches"):
        ReanalysisPipeline(
            in_channels=4,
            action_space_size=2,
            search_config=SearchConfig(num_simulations=1),
            policy_chunk_size=4,
            cache_targets=True,
            cache_target_ttl=200,
            rng_seed=0,
            support_min=-300,
            support_max=300,
            precision="fp32",
            search_threads=1,
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
