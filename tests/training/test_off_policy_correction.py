from dataclasses import replace

import pytest
import torch

from atariagent.replay_batch import ReplayBatch


def _batch() -> ReplayBatch:
    return ReplayBatch(
        frames=torch.zeros(2, 2, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 1, 1, dtype=torch.long),
        rewards=torch.zeros(2, 1),
        policy_targets=torch.full((2, 2, 2), 0.5),
        value_targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        policy_mask=torch.tensor([[True, True], [True, False]]),
        value_mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.tensor([10, 11]),
        importance_weights=torch.ones(2),
        search_value_targets=torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        transition_ages=torch.tensor([5_000, 4_999]),
    )


def test_mixed_values_use_td_before_start_step() -> None:
    batch = _batch()

    selected = batch.with_selected_value_targets(
        mode="mixed",
        learner_step=29_999,
        collection_steps=100_000,
        mixed_start_step=30_000,
        freshness_threshold=5_000,
    )

    torch.testing.assert_close(selected.value_targets, batch.value_targets)


def test_mixed_values_use_search_for_stale_samples_at_strict_boundary() -> None:
    batch = _batch()

    selected = batch.with_selected_value_targets(
        mode="mixed",
        learner_step=30_000,
        collection_steps=100_000,
        mixed_start_step=30_000,
        freshness_threshold=5_000,
    )

    torch.testing.assert_close(
        selected.value_targets,
        torch.tensor([[10.0, 20.0], [3.0, 4.0]]),
    )


def test_mixed_values_preserve_transition_ages_through_final_updates() -> None:
    batch = replace(
        _batch(),
        transition_ages=torch.tensor([0, 4_999]),
    )

    first_final_update = batch.with_selected_value_targets(
        mode="mixed",
        learner_step=100_001,
        collection_steps=100_000,
        mixed_start_step=30_000,
        freshness_threshold=5_000,
    )
    after_threshold = batch.with_selected_value_targets(
        mode="mixed",
        learner_step=105_000,
        collection_steps=100_000,
        mixed_start_step=30_000,
        freshness_threshold=5_000,
    )

    torch.testing.assert_close(
        first_final_update.value_targets,
        batch.value_targets,
    )
    torch.testing.assert_close(
        after_threshold.value_targets,
        batch.value_targets,
    )


def test_mixed_values_preserve_stale_boundary_during_final_updates() -> None:
    batch = _batch()
    selected = batch.with_selected_value_targets(
        mode="mixed",
        learner_step=105_000,
        collection_steps=100_000,
        mixed_start_step=30_000,
        freshness_threshold=5_000,
    )

    torch.testing.assert_close(
        selected.value_targets,
        torch.tensor([[10.0, 20.0], [3.0, 4.0]]),
    )


def test_search_values_fall_back_to_td_without_a_valid_search_root() -> None:
    batch = _batch()

    selected = batch.with_selected_value_targets(
        mode="search",
        learner_step=0,
        collection_steps=100_000,
        mixed_start_step=30_000,
        freshness_threshold=5_000,
    )

    torch.testing.assert_close(
        selected.value_targets,
        torch.tensor([[10.0, 20.0], [30.0, 4.0]]),
    )


@pytest.mark.parametrize(
    ("mode", "learner_step", "expected_targets", "expected_mask"),
    [
        ("td", 30_000, [[1.0, 2.0], [3.0, 4.0]], [[False, False], [False, False]]),
        ("mixed", 29_999, [[1.0, 2.0], [3.0, 4.0]], [[False, False], [False, False]]),
        ("mixed", 30_000, [[10.0, 20.0], [3.0, 4.0]], [[True, True], [False, False]]),
        ("search", 0, [[10.0, 20.0], [30.0, 4.0]], [[True, True], [True, False]]),
    ],
)
def test_search_target_validity_does_not_require_td_validity(
    mode, learner_step, expected_targets, expected_mask,
) -> None:
    batch = replace(_batch(), value_mask=torch.zeros(2, 2, dtype=torch.bool))
    selected = batch.with_selected_value_targets(
        mode=mode,
        learner_step=learner_step,
        collection_steps=100_000,
        mixed_start_step=30_000,
        freshness_threshold=5_000,
    )
    torch.testing.assert_close(selected.value_targets, torch.tensor(expected_targets))
    torch.testing.assert_close(selected.value_mask, torch.tensor(expected_mask))
    assert not batch.value_mask.any()  # Selection must not mutate the source batch.


def test_mixed_value_selection_requires_reanalysis_metadata() -> None:
    batch = replace(_batch(), search_value_targets=None)

    with pytest.raises(ValueError, match="search value"):
        batch.with_selected_value_targets(
            mode="mixed",
            learner_step=30_000,
            collection_steps=100_000,
            mixed_start_step=30_000,
            freshness_threshold=5_000,
        )
