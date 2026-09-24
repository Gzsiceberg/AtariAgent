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
    reanalysis_frames = torch.zeros(
        batch_size, 6, 1, 96, 96, dtype=torch.uint8
    )
    return ReplayBatch(
        frames=reanalysis_frames[:, :5],
        actions=torch.zeros(batch_size, 1, 1, dtype=torch.long),
        rewards=torch.zeros(batch_size, 1),
        policy_targets=torch.full((batch_size, 2, 2), 0.5),
        value_targets=torch.zeros(batch_size, 2),
        action_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        value_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size),
        value_bootstrap_frames=reanalysis_frames[:, 1:],
        reanalysis_frames=reanalysis_frames,
        value_bootstrap_values=torch.ones(batch_size, 2),
        value_bootstrap_discounts=torch.ones(batch_size, 2),
        value_bootstrap_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        value_bootstrap_state_ids=(
            torch.arange(batch_size * 2).view(batch_size, 2) + 100
        ),
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
    cache_targets: bool = True,
    cache_target_ttl: int = 200,
    action_embedding: bool = True,
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
        cache_targets=cache_targets,
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
        action_embedding=action_embedding,
    )
    representation = RepresentationNetwork(4)
    prediction = PredictionNetwork(2)
    dynamics = DynamicsNetwork(2, action_embedding=action_embedding)
    pipeline.publish_weights(
        0,
        make_target_state(representation, prediction, dynamics),
        wait=True,
    )
    return pipeline


@pytest.mark.parametrize("action_embedding", [True, False])
@pytest.mark.parametrize("search_algorithm", ["puct", "gumbel"])
def test_native_pipeline_enforces_prefetch_bound_and_matches_requests(
    action_embedding, search_algorithm
) -> None:
    pipeline = _pipeline(
        prefetch_batches=2,
        action_embedding=action_embedding,
        search_algorithm=search_algorithm,
    )
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
        policy_targets=torch.tensor([0.9, 0.1]).expand_as(batch.policy_targets).clone(),
    )
    try:
        pipeline.submit(batch, trained_steps=0)
        reanalyzed = pipeline.wait_next().batch

        assert not torch.equal(reanalyzed.policy_targets, batch.policy_targets)
    finally:
        pipeline.close()


def test_native_reanalysis_skips_zero_policy_targets() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    batch = _batch()
    targets = batch.policy_targets.clone()
    targets[:, 1] = 0
    batch = replace(batch, policy_targets=targets)
    try:
        pipeline.submit(batch)
        ready = pipeline.wait_next()
        assert ready.policy_roots_searched == 2
        torch.testing.assert_close(
            ready.batch.policy_mask, torch.tensor([[True, False], [True, False]])
        )
        assert not ready.batch.policy_targets[:, 1].any()
        assert not ready.batch.search_value_targets[:, 1].any()
    finally:
        pipeline.close()


def test_native_pipeline_uses_consolidated_reanalysis_frames() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    batch = _batch()
    separate_batch = replace(
        batch,
        frames=batch.frames.clone(),
        value_bootstrap_frames=batch.value_bootstrap_frames.clone(),
        reanalysis_frames=None,
    )
    assert replay_batch_nbytes(batch) < replay_batch_nbytes(separate_batch)
    try:
        pipeline.submit(batch)
        ready = pipeline.wait_next()
        assert ready.batch.search_value_targets is not None
        assert ready.policy_roots_searched == 3
    finally:
        pipeline.close()


def test_cache_does_not_copy_an_invalid_value_to_a_valid_duplicate() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    batch = replace(
        _batch(),
        policy_targets=torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.5, 0.5], [0.0, 0.0]]]),
        value_mask=torch.tensor([[False, False], [True, False]]),
        value_targets=torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
        value_bootstrap_mask=torch.zeros(2, 2, dtype=torch.bool),
        reanalysis_state_ids=torch.tensor([[7, -1], [7, -1]]),
    )
    try:
        pipeline.submit(batch)
        ready = pipeline.wait_next()

        assert ready.policy_roots_searched == 1
        assert ready.batch.value_targets[1, 0] == 10.0
    finally:
        pipeline.close()


def test_value_cache_reuses_raw_predictions_for_occurrence_specific_targets() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    batch = replace(
        _batch(),
        value_targets=torch.tensor([[10.0, 0.0], [20.0, 0.0]]),
        value_bootstrap_values=torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
        value_bootstrap_discounts=torch.tensor([[1.0, 0.0], [0.5, 0.0]]),
        value_bootstrap_mask=torch.tensor(
            [[True, False], [True, False]]
        ),
        value_bootstrap_state_ids=torch.tensor([[9, -1], [9, -1]]),
    )
    try:
        pipeline.submit(batch)
        first = pipeline.wait_next()
        pipeline.submit(batch)
        cached = pipeline.wait_next()

        assert first.value_roots_requested == 2
        assert first.value_roots_searched == 1
        assert cached.value_roots_searched == 0
        assert cached.value_cache_hits == 2
        torch.testing.assert_close(
            first.batch.value_targets[1, 0],
            14.0 + 0.5 * first.batch.value_targets[0, 0],
        )
        torch.testing.assert_close(
            cached.batch.value_targets,
            first.batch.value_targets,
        )
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
        assert first.value_roots_requested == 4
        assert first.value_roots_searched == 4
        assert first.value_cache_hits == 0
        assert cached.value_roots_requested == 4
        assert cached.value_roots_searched == 0
        assert cached.value_cache_hits == 4
        assert pipeline.value_cache_size == 4
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
        assert refreshed.value_roots_searched == 0
        assert refreshed.value_cache_hits == 4
        assert pipeline.cache_size == 3
        assert pipeline.value_cache_size == 4

        representation = RepresentationNetwork(4)
        prediction = PredictionNetwork(2)
        dynamics = DynamicsNetwork(2)
        pipeline.publish_weights(
            1,
            make_target_state(representation, prediction, dynamics),
        )
        assert pipeline.cache_size == 0
        assert pipeline.value_cache_size == 0
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
        assert expired.policy_roots_searched == 3
        assert expired.cache_hits == 0
        assert expired.value_roots_searched == 0
        assert expired.value_cache_hits == 4
    finally:
        pipeline.close()


def test_native_pipeline_validates_trained_steps() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    try:
        for trained_steps in (True, 1.0):
            with pytest.raises(TypeError, match="trained_steps must be an integer"):
                pipeline.submit(
                    _batch(), trained_steps=trained_steps  # type: ignore[arg-type]
                )
        with pytest.raises(ValueError, match="trained_steps must be non-negative"):
            pipeline.submit(_batch(), trained_steps=-1)
    finally:
        pipeline.close()


def test_native_pipeline_validates_native_batch_tensor_contract() -> None:
    pipeline = _pipeline(prefetch_batches=1)
    try:
        with pytest.raises(ValueError, match="policy_mask must be a 2D bool"):
            pipeline.submit(
                replace(_batch(), policy_targets=torch.ones(2, 2))
            )
        with pytest.raises(ValueError, match="indices must be a 1D int64"):
            pipeline.submit(
                replace(_batch(), indices=torch.arange(2, dtype=torch.int32))
            )
        with pytest.raises(ValueError, match="reanalysis_frames"):
            pipeline.submit(replace(_batch(), reanalysis_frames=None))
        for field_name in (
            "value_bootstrap_frames",
            "value_bootstrap_values",
            "value_bootstrap_discounts",
            "value_bootstrap_mask",
            "value_bootstrap_state_ids",
        ):
            with pytest.raises(ValueError, match=field_name):
                pipeline.submit(replace(_batch(), **{field_name: None}))
        with pytest.raises(ValueError, match="reanalysis_state_ids"):
            pipeline.submit(replace(_batch(), reanalysis_state_ids=None))
        with pytest.raises(ValueError, match="reanalysis_state_ids"):
            pipeline.submit(
                replace(
                    _batch(),
                    reanalysis_state_ids=torch.zeros(2, 2, dtype=torch.int32),
                )
            )
        for field_name in (
            "value_bootstrap_values",
            "value_bootstrap_discounts",
        ):
            for invalid in (
                torch.zeros(1, 2),
                torch.zeros(2),
                torch.zeros(2, 2, dtype=torch.float64),
            ):
                with pytest.raises(ValueError, match=field_name):
                    pipeline.submit(
                        replace(_batch(), **{field_name: invalid})
                    )
    finally:
        pipeline.close()


def test_native_pipeline_accepts_noncontiguous_bootstrap_terms() -> None:
    pipeline = _pipeline(prefetch_batches=1, cache_targets=False)
    batch = replace(
        _batch(),
        value_bootstrap_values=torch.ones(2, 4)[:, ::2],
        value_bootstrap_discounts=torch.ones(2, 4)[:, ::2],
    )
    assert not batch.value_bootstrap_values.is_contiguous()
    assert not batch.value_bootstrap_discounts.is_contiguous()
    try:
        pipeline.submit(batch)
        ready = pipeline.wait_next()
        assert ready.batch.value_targets.shape == batch.value_targets.shape
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
