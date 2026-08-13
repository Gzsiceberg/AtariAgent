from dataclasses import replace

import numpy as np
import pytest
import torch

from atariagent.replay import FIFOReplayBuffer, ReplayBatch
from atariagent.search import SearchResult
from atariagent.selfplay import GameTrajectory


def make_trajectory(
    length: int,
    *,
    block_id: int = 0,
    episode_id: int = 0,
    initial_value: int = 0,
    stack_size: int = 1,
    terminated: bool = False,
    truncated: bool = False,
    lookahead_steps: int = 0,
) -> GameTrajectory:
    frames = tuple(
        np.full((1, 2, 2), initial_value + step, dtype=np.uint8)
        for step in range(length + stack_size)
    )
    results = tuple(
        SearchResult(
            action=step % 2,
            visit_counts=(1, 3),
            root_value=float(initial_value + step),
        )
        for step in range(length)
    )
    return GameTrajectory(
        environment_index=0,
        episode_id=episode_id,
        block_id=block_id,
        stack_size=stack_size,
        frames=frames,
        actions=tuple(step % 2 for step in range(length)),
        rewards=tuple(float(step + 1) for step in range(length)),
        raw_rewards=tuple(float(step + 1) for step in range(length)),
        search_results=results,
        terminated=terminated,
        truncated=truncated,
        full_episode_done=terminated or truncated,
        lookahead_steps=lookahead_steps,
    )


def test_fifo_replay_evicts_oldest_complete_trajectories() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=5, unroll_steps=2, td_steps=2
    )

    replay.add(make_trajectory(3, block_id=0))
    result = replay.add(make_trajectory(3, block_id=1, initial_value=10))

    assert len(replay) == 3
    assert replay.trajectory_count == 1
    assert replay.utilization == pytest.approx(0.6)
    assert result.evicted_trajectories == 1
    assert result.evicted_transitions == 3

    batch = replay.sample(3)
    assert torch.all(batch.frames[:, 0] >= 10)
    final_sample = int(
        (batch.frames[:, 0, 0, 0, 0] == 12).nonzero().item()
    )
    torch.testing.assert_close(
        batch.policy_mask[final_sample],
        torch.tensor([True, False, False]),
    )


def test_replay_samples_padded_five_step_tensor_batches() -> None:
    replay = FIFOReplayBuffer(max_transitions=10, seed=3)
    replay.add(make_trajectory(3, terminated=True))

    batch = replay.sample(batch_size=3)

    assert batch.frames.shape == (3, 6, 1, 2, 2)
    assert batch.frames.dtype == torch.uint8
    assert batch.actions.shape == (3, 5, 1)
    assert batch.actions.dtype == torch.long
    assert batch.rewards.shape == (3, 5)
    assert batch.policy_targets.shape == (3, 6, 2)
    assert batch.value_targets.shape == (3, 6)
    assert batch.action_mask.dtype == torch.bool
    assert batch.policy_mask.dtype == torch.bool
    assert batch.value_mask.dtype == torch.bool
    assert batch.value_bootstrap_frames is not None
    assert batch.value_bootstrap_frames.shape == batch.frames.shape
    assert batch.value_bootstrap_values is not None
    assert batch.value_bootstrap_values.shape == (3, 6)
    assert batch.value_bootstrap_discounts is not None
    assert batch.value_bootstrap_discounts.shape == (3, 6)
    assert batch.value_bootstrap_mask is not None
    assert batch.value_bootstrap_mask.shape == (3, 6)
    assert batch.indices.shape == (3,)
    assert batch.indices.dtype == torch.long
    assert batch.importance_weights.shape == (3,)
    assert batch.importance_weights.dtype == torch.float32
    assert batch.importance_weights.max() == pytest.approx(1.0)

    sampled_start_values = batch.frames[:, 0, 0, 0, 0]
    assert sorted(sampled_start_values.tolist()) == [0, 1, 2]
    final_sample = int((sampled_start_values == 2).nonzero().item())
    torch.testing.assert_close(
        batch.action_mask[final_sample],
        torch.tensor([True, False, False, False, False]),
    )
    torch.testing.assert_close(
        batch.policy_mask[final_sample],
        torch.tensor([True, False, False, False, False, False]),
    )
    assert batch.value_targets[final_sample, 0] == 3.0
    assert batch.value_targets[final_sample, 1] == 0.0
    assert torch.all(batch.value_mask[final_sample, :2])
    assert torch.all(batch.policy_targets[final_sample, 1] == 0)
    assert batch.rewards[final_sample, 0] == 3.0
    assert torch.all(batch.rewards[final_sample, 1:] == 0)
    assert torch.all(batch.frames[final_sample, 1:] == 3)
    torch.testing.assert_close(
        batch.policy_targets[final_sample, 0], torch.tensor([0.25, 0.75])
    )
    assert torch.all(batch.policy_targets[final_sample, 1:] == 0)

    normalized = batch.normalized_observations()
    assert normalized.dtype == torch.float32
    torch.testing.assert_close(
        normalized[final_sample, 0],
        torch.full((1, 2, 2), 2.0 / 255.0),
    )


def test_replay_can_skip_target_network_bootstrap_metadata() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=2,
        td_steps=2,
        seed=3,
    )
    replay.add(make_trajectory(3, terminated=True))

    batch = replay.sample(
        batch_size=3,
        include_value_bootstraps=False,
    )

    assert batch.value_bootstrap_frames is None
    assert batch.value_bootstrap_values is None
    assert batch.value_bootstrap_discounts is None
    assert batch.value_bootstrap_mask is None


def test_replay_builds_fixed_n_step_values_from_stored_root_values() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=3,
        td_steps=2,
        discount=0.5,
        seed=3,
    )
    replay.add(make_trajectory(3, terminated=True))

    batch = replay.sample(batch_size=3)
    start_zero = int((batch.frames[:, 0, 0, 0, 0] == 0).nonzero().item())

    torch.testing.assert_close(
        batch.value_targets[start_zero],
        torch.tensor([2.5, 3.5, 3.0, 0.0]),
    )
    torch.testing.assert_close(
        batch.value_mask[start_zero],
        torch.tensor([True, True, True, True]),
    )
    assert batch.value_bootstrap_mask is not None
    assert batch.value_bootstrap_values is not None
    assert batch.value_bootstrap_discounts is not None
    torch.testing.assert_close(
        batch.value_bootstrap_mask[start_zero],
        torch.tensor([True, False, False, False]),
    )
    assert batch.value_bootstrap_values[start_zero, 0] == 2.0
    assert batch.value_bootstrap_discounts[start_zero, 0] == 0.25
    bootstrap = batch.normalized_value_bootstrap_observation(0)
    torch.testing.assert_close(
        bootstrap[start_zero],
        torch.full((1, 2, 2), 2.0 / 255.0),
    )

    fresh_bootstraps = torch.zeros_like(batch.value_targets)
    fresh_bootstraps[start_zero, 0] = 10.0
    reanalyzed = batch.with_reanalyzed_value_targets(fresh_bootstraps)
    assert reanalyzed.value_targets[start_zero, 0] == 4.5
    torch.testing.assert_close(
        reanalyzed.value_targets[start_zero, 1:],
        batch.value_targets[start_zero, 1:],
    )


def test_replay_builds_value_targets_from_local_lookahead() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=6,
        unroll_steps=5,
        td_steps=5,
        discount=0.5,
    )
    replay.add(make_trajectory(11, lookahead_steps=5))

    trajectory = replay._trajectories[0]
    expected = np.asarray(
        [
            sum(0.5**step * (offset + step + 1) for step in range(5))
            + 0.5**5 * (offset + 5)
            for offset in range(6)
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(trajectory.value_targets[:6], expected)
    np.testing.assert_array_equal(
        trajectory.value_valid_mask[:6], np.ones(6, dtype=bool)
    )
    np.testing.assert_array_equal(
        trajectory.root_values[5:11], np.arange(5, 11, dtype=np.float32)
    )


def test_replay_requires_lookahead_to_cover_unroll_and_td_steps() -> None:
    trajectory = make_trajectory(5, lookahead_steps=2)

    with pytest.raises(ValueError, match="lookahead_steps"):
        FIFOReplayBuffer(
            max_transitions=3, unroll_steps=3, td_steps=2
        ).add(trajectory)
    with pytest.raises(ValueError, match="lookahead_steps"):
        FIFOReplayBuffer(
            max_transitions=3, unroll_steps=2, td_steps=3
        ).add(trajectory)

    replay = FIFOReplayBuffer(
        max_transitions=3, unroll_steps=2, td_steps=2
    )
    replay.add(trajectory)
    assert replay.sample(batch_size=1).action_mask.all()


def test_replay_reconstructs_overlapping_stacks_from_compact_frames() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10, unroll_steps=1, td_steps=1, seed=1
    )
    replay.add(make_trajectory(2, stack_size=4))

    batch = replay.sample(batch_size=2)

    assert batch.frames.shape == (2, 5, 1, 2, 2)
    observations = batch.stacked_observations()
    assert observations.shape == (2, 2, 4, 2, 2)
    first_sample = int((batch.frames[:, 0, 0, 0, 0] == 0).nonzero().item())
    torch.testing.assert_close(
        observations[first_sample, 0, :, 0, 0],
        torch.tensor([0, 1, 2, 3], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        observations[first_sample, 1, :, 0, 0],
        torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
    )


@pytest.mark.parametrize(
    ("channels", "stack_size", "unroll_steps"),
    [(1, 1, 1), (1, 4, 3), (3, 4, 5)],
)
def test_root_normalization_exactly_matches_full_root_slice(
    channels: int, stack_size: int, unroll_steps: int
) -> None:
    batch_size = 2
    states = unroll_steps + 1
    frames = torch.randint(
        0,
        256,
        (batch_size, stack_size + unroll_steps, channels, 3, 2),
        dtype=torch.uint8,
    )
    batch = ReplayBatch(
        frames=frames,
        actions=torch.zeros(batch_size, unroll_steps, 1, dtype=torch.long),
        rewards=torch.zeros(batch_size, unroll_steps),
        policy_targets=torch.zeros(batch_size, states, 2),
        value_targets=torch.zeros(batch_size, states),
        action_mask=torch.ones(batch_size, unroll_steps, dtype=torch.bool),
        policy_mask=torch.zeros(batch_size, states, dtype=torch.bool),
        value_mask=torch.ones(batch_size, states, dtype=torch.bool),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size),
    )

    root = batch.normalized_root_observation()

    assert root.shape == (batch_size, stack_size * channels, 3, 2)
    assert root.dtype == torch.float32
    assert root.device == frames.device
    assert 0.0 <= root.min() <= root.max() <= 1.0
    torch.testing.assert_close(root, batch.normalized_observations()[:, 0])


def test_replay_unroll_uses_local_lookahead_transitions() -> None:
    trajectory = make_trajectory(7, lookahead_steps=5)
    replay = FIFOReplayBuffer(
        max_transitions=10, unroll_steps=3, td_steps=3, seed=2
    )
    replay.add(trajectory)

    batch = replay.sample(batch_size=2)
    crossing_sample = int(
        (batch.frames[:, 0, 0, 0, 0] == 1).nonzero().item()
    )

    torch.testing.assert_close(
        batch.frames[crossing_sample, :, 0, 0, 0],
        torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        batch.action_mask[crossing_sample],
        torch.tensor([True, True, True]),
    )
    torch.testing.assert_close(
        batch.policy_mask[crossing_sample],
        torch.tensor([True, True, True, True]),
    )
    value_replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=3,
        td_steps=2,
        discount=0.5,
        seed=2,
    )
    value_replay.add(trajectory)
    value_batch = value_replay.sample(batch_size=2)
    crossing_value_sample = int(
        (value_batch.frames[:, 0, 0, 0, 0] == 1).nonzero().item()
    )
    torch.testing.assert_close(
        value_batch.value_targets[crossing_value_sample],
        torch.tensor([4.25, 6.0, 7.75, 9.5]),
    )
    torch.testing.assert_close(
        value_batch.value_mask[crossing_value_sample],
        torch.tensor([True, True, True, True]),
    )
    assert value_batch.value_bootstrap_values is not None
    assert value_batch.value_bootstrap_discounts is not None
    assert value_batch.value_bootstrap_mask is not None
    torch.testing.assert_close(
        value_batch.value_bootstrap_values[crossing_value_sample],
        torch.tensor([3.0, 4.0, 5.0, 6.0]),
    )
    torch.testing.assert_close(
        value_batch.value_bootstrap_discounts[crossing_value_sample],
        torch.full((4,), 0.25),
    )
    assert value_batch.value_bootstrap_mask[crossing_value_sample].all()
    for offset in range(4):
        bootstrap = value_batch.normalized_value_bootstrap_observation(offset)
        torch.testing.assert_close(
            bootstrap[crossing_value_sample],
            torch.full((1, 2, 2), (3.0 + offset) / 255.0),
        )


def reference_value_targets(
    trajectory: GameTrajectory,
    position: int,
    *,
    target_count: int,
    td_steps: int,
    discount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros(target_count, dtype=np.float32)
    valid = np.zeros(target_count, dtype=bool)
    bootstrap_mask = np.zeros(target_count, dtype=bool)
    stored_count = trajectory.stored_transition_count
    for offset in range(target_count):
        target_position = position + offset
        if target_position == stored_count and trajectory.terminated:
            valid[offset] = True
            continue
        if target_position >= stored_count:
            continue
        bootstrap_position = target_position + td_steps
        reward_end = min(bootstrap_position, stored_count)
        reward_return = sum(
            discount**step * trajectory.rewards[target_position + step]
            for step in range(reward_end - target_position)
        )
        if bootstrap_position < stored_count:
            reward_return += (
                discount**td_steps
                * trajectory.search_results[bootstrap_position].root_value
            )
            valid[offset] = True
            bootstrap_mask[offset] = True
            values[offset] = reward_return
        elif trajectory.terminated:
            valid[offset] = True
            values[offset] = reward_return
    return values, valid, bootstrap_mask


@pytest.mark.parametrize(
    "trajectory",
    [
        make_trajectory(4, terminated=True),
        make_trajectory(4, truncated=True),
        make_trajectory(7, lookahead_steps=3),
        make_trajectory(4),
    ],
)
def test_precomputed_values_match_reference_for_boundary_types(
    trajectory: GameTrajectory,
) -> None:
    replay = FIFOReplayBuffer(
        10,
        unroll_steps=2,
        td_steps=3,
        discount=0.5,
        seed=5,
    )
    replay.add(trajectory)
    batch = replay.sample(len(trajectory))
    assert batch.value_bootstrap_mask is not None

    for batch_index in range(batch.batch_size):
        position = int(batch.frames[batch_index, 0, 0, 0, 0])
        expected_values, expected_valid, expected_bootstrap = (
            reference_value_targets(
                trajectory,
                position,
                target_count=3,
                td_steps=3,
                discount=0.5,
            )
        )
        np.testing.assert_allclose(
            batch.value_targets[batch_index].numpy(), expected_values
        )
        np.testing.assert_array_equal(
            batch.value_mask[batch_index].numpy(), expected_valid
        )
        np.testing.assert_array_equal(
            batch.value_bootstrap_mask[batch_index].numpy(),
            expected_bootstrap,
        )


def test_prioritized_replay_samples_and_updates_efficientzero_priorities() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=1,
        td_steps=1,
        seed=4,
        priority_alpha=0.6,
    )
    replay.add(make_trajectory(3, terminated=True))
    replay.update_priorities(
        np.array([0, 1, 2]), np.array([1.0, 4.0, 16.0])
    )

    batch = replay.sample(batch_size=3, priority_beta=0.4)

    probabilities = np.array([1.0, 4.0, 16.0]) ** 0.6
    probabilities /= probabilities.sum()
    expected_weights = (3 * probabilities[batch.indices.numpy()]) ** -0.4
    expected_weights /= expected_weights.max()
    np.testing.assert_allclose(
        batch.importance_weights.numpy(), expected_weights, rtol=1e-6
    )

    counts = np.zeros(3, dtype=np.int64)
    for _ in range(300):
        sampled = replay.sample(1, priority_beta=0.4)
        counts[int(sampled.indices.item())] += 1
    assert counts[2] > counts[1] > counts[0]


def test_replay_sample_batch_applies_configured_priority_beta_schedule() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=1,
        td_steps=1,
        priority_beta_initial=0.4,
        priority_beta_final=1.0,
    )
    replay.add(make_trajectory(3, terminated=True))

    batch, beta = replay.sample_batch(
        2,
        trained_steps=60_000,
        training_steps=120_000,
        include_value_bootstraps=False,
        pin_memory=False,
    )

    assert beta == pytest.approx(0.7)
    assert batch.value_bootstrap_frames is None
    assert replay.priority_beta(0, 120_000) == pytest.approx(0.4)
    assert replay.priority_beta(120_000, 120_000) == pytest.approx(1.0)
    assert replay.priority_beta(130_000, 120_000) == pytest.approx(1.0)


def test_new_replay_transitions_receive_current_max_priority() -> None:
    replay = FIFOReplayBuffer(max_transitions=10)
    replay.add(make_trajectory(2, terminated=True))
    replay.update_priorities([0, 1], [2.0, 5.0])

    replay.add(
        make_trajectory(
            1,
            episode_id=1,
            initial_value=10,
            terminated=True,
        )
    )

    np.testing.assert_allclose(replay.priorities, np.array([2.0, 5.0, 5.0]))


def test_replay_rejects_invalid_capacity_and_oversized_samples() -> None:
    with pytest.raises(ValueError, match="positive"):
        FIFOReplayBuffer(0)

    replay = FIFOReplayBuffer(3)
    with pytest.raises(ValueError, match="exceeds"):
        replay.add(make_trajectory(4))

    replay.add(make_trajectory(2))
    with pytest.raises(ValueError, match="cannot sample"):
        replay.sample(3)


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    [
        ("unroll_steps", 0, ValueError),
        ("unroll_steps", True, TypeError),
        ("td_steps", 0, ValueError),
        ("td_steps", 1.5, TypeError),
        ("discount", -0.1, ValueError),
        ("discount", 1.1, ValueError),
        ("discount", float("nan"), ValueError),
        ("priority_beta_initial", -0.1, ValueError),
        ("priority_beta_initial", 1.1, ValueError),
        ("priority_beta_final", 1.1, ValueError),
        ("priority_beta_final", float("nan"), ValueError),
    ],
)
def test_replay_validates_fixed_target_configuration(
    keyword: str, value: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        FIFOReplayBuffer(3, **{keyword: value})  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_replay_can_construct_batches_in_pinned_memory() -> None:
    replay = FIFOReplayBuffer(3, unroll_steps=1, td_steps=1)
    replay.add(make_trajectory(2, terminated=True))

    batch = replay.sample(2, pin_memory=True)

    assert batch.frames.is_pinned()
    assert batch.actions.is_pinned()
    assert batch.value_targets.is_pinned()
    assert batch.indices.is_pinned()


def test_replay_sample_does_not_accept_target_configuration_overrides() -> None:
    replay = FIFOReplayBuffer(3, unroll_steps=1, td_steps=1, discount=0.5)
    replay.add(make_trajectory(1, terminated=True))

    assert replay.unroll_steps == 1
    assert replay.td_steps == 1
    assert replay.discount == 0.5
    with pytest.raises(AttributeError):
        replay.unroll_steps = 2  # type: ignore[misc]
    with pytest.raises(TypeError, match="unexpected keyword"):
        replay.sample(1, unroll_steps=2)  # type: ignore[call-arg]


def test_replay_prepares_read_only_contiguous_targets_on_add() -> None:
    replay = FIFOReplayBuffer(3, unroll_steps=2, td_steps=2, discount=0.5)
    replay.add(make_trajectory(3, terminated=True))

    stored = replay._trajectories[0]
    for array in (
        stored.frames,
        stored.actions,
        stored.rewards,
        stored.policy_targets,
        stored.root_values,
        stored.value_targets,
        stored.value_valid_mask,
    ):
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    np.testing.assert_allclose(stored.policy_targets, [[0.25, 0.75]] * 3)
    np.testing.assert_allclose(stored.value_targets, [2.5, 3.5, 3.0, 0.0])
    np.testing.assert_array_equal(stored.value_valid_mask, np.ones(4, bool))


def test_failed_trajectory_preparation_does_not_evict_existing_data() -> None:
    replay = FIFOReplayBuffer(2, unroll_steps=1, td_steps=1, seed=1)
    replay.add(make_trajectory(2, terminated=True))
    invalid = make_trajectory(
        2,
        episode_id=1,
        initial_value=10,
        terminated=True,
    )
    invalid_result = SearchResult(
        action=0,
        visit_counts=(0, 0),
        root_value=0.0,
    )
    invalid = replace(
        invalid,
        search_results=(invalid_result,) * invalid.stored_transition_count,
    )

    with pytest.raises(ValueError, match="visited actions"):
        replay.add(invalid)

    assert len(replay) == 2
    assert replay.trajectory_count == 1
    batch = replay.sample(2)
    assert sorted(batch.frames[:, 0, 0, 0, 0].tolist()) == [0, 1]
