from dataclasses import fields, replace

import pytest
import torch

from atariagent.replay_batch import ReplayBatch


def _batch() -> ReplayBatch:
    return ReplayBatch(
        frames=torch.zeros(2, 2, 1, 1, 1, dtype=torch.uint8),
        actions=torch.zeros(2, 1, 1, dtype=torch.long),
        rewards=torch.zeros(2, 1),
        policy_targets=torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.0, 0.0]]]),
        value_targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        action_mask=torch.ones(2, 1, dtype=torch.bool),
        indices=torch.tensor([10, 11]),
        importance_weights=torch.ones(2),
        search_value_targets=torch.tensor([[10.0, 20.0], [30.0, 0.0]]),
        transition_ages=torch.tensor([4_999, 4_998]),
    )


def test_policy_validity_is_derived_from_zero_targets_not_stored() -> None:
    batch = _batch()
    assert "policy_mask" not in {field.name for field in fields(batch)}
    torch.testing.assert_close(
        batch.policy_mask, torch.tensor([[True, True], [True, False]])
    )
    padded = replace(batch, policy_targets=torch.zeros_like(batch.policy_targets))
    assert not padded.policy_mask.any()
    assert batch.policy_mask.any()


def test_zero_policy_targets_have_zero_loss_and_gradient() -> None:
    targets = _batch().policy_targets
    logits = torch.randn(4, 2, requires_grad=True)
    losses = ReplayBatch._policy_cross_entropy(logits, targets.flatten(0, 1))
    assert losses[-1] == 0
    losses.sum().backward()
    assert torch.count_nonzero(logits.grad[-1]) == 0
    assert torch.count_nonzero(logits.grad[:-1]) > 0


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
        transition_ages=torch.tensor([0, 4_998]),
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


def test_search_values_use_zero_at_block_endpoint_not_td_fallback() -> None:
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
        torch.tensor([[10.0, 20.0], [30.0, 0.0]]),
    )


@pytest.mark.parametrize("mode", ["td", "mixed", "search"])
def test_value_loss_mask_requires_an_original_block_policy_target(mode) -> None:
    batch = replace(_batch(), action_mask=torch.tensor([[False], [True]]))
    selected = batch.with_selected_value_targets(
        mode=mode,
        learner_step=30_000,
        collection_steps=100_000,
        mixed_start_step=30_000,
        freshness_threshold=5_000,
    )
    assert "value_mask" not in {field.name for field in fields(batch)}
    torch.testing.assert_close(selected.value_mask, torch.tensor([[True, False], [True, False]]))
    torch.testing.assert_close(selected.value_mask, batch.value_mask)


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
