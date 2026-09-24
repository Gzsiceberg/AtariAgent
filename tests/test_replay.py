from dataclasses import fields, replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from atariagent.replay import FIFOReplayBuffer
from atariagent.search import SearchResult
from atariagent.selfplay import GameTrajectory


def make_trajectory(
    length: int,
    *,
    block_id: int = 0,
    episode_id: int = 0,
    environment_index: int = 0,
    initial_value: int = 0,
    stack_size: int = 1,
    terminated: bool = False,
    truncated: bool = False,
    lookahead_steps: int = 0,
    predicted_values: tuple[float, ...] = (),
) -> GameTrajectory:
    frames = tuple(
        np.full((1, 2, 2), initial_value + step, dtype=np.uint8)
        for step in range(length + stack_size)
    )
    results = tuple(
        SearchResult(
            action=step % 2,
            policy_target=(0.25, 0.75),
            root_value=float(initial_value + step),
        )
        for step in range(length)
    )
    if not predicted_values:
        predicted_values = tuple(
            float(initial_value + step) for step in range(length)
        )
    return GameTrajectory(
        environment_index=environment_index,
        episode_id=episode_id,
        block_id=block_id,
        stack_size=stack_size,
        frames=frames,
        actions=tuple(step % 2 for step in range(length)),
        rewards=tuple(float(step + 1) for step in range(length)),
        raw_rewards=tuple(float(step + 1) for step in range(length)),
        search_results=results,
        predicted_values=predicted_values,
        terminated=terminated,
        truncated=truncated,
        full_episode_done=terminated or truncated,
        lookahead_steps=lookahead_steps,
    )


@pytest.mark.parametrize("tensor_arrays", [False, True])
def test_replay_state_round_trip_restores_data_priorities_and_rng(
    tmp_path: Path, tensor_arrays: bool,
) -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=2,
        discount=0.5,
        seed=7,
    )
    replay.add(make_trajectory(3, block_id=0, terminated=True))
    replay.add(
        make_trajectory(
            3,
            block_id=1,
            episode_id=1,
            initial_value=10,
            terminated=True,
        )
    )
    replay.update_priorities([0, 1], [2.0, 4.0])

    restored = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=2,
        discount=0.5,
        seed=999,
    )
    state = replay.state_dict(tensor_arrays=tensor_arrays)
    original_state = replay.state_dict()
    for trajectory, original in zip(
        state["trajectories"], original_state["trajectories"], strict=True,
    ):
        assert "initial_priority" not in trajectory
        assert "initial_priorities" in trajectory
        for name, value in original.items():
            if isinstance(value, np.ndarray):
                encoded = trajectory[name]
                if tensor_arrays:
                    assert isinstance(encoded, torch.Tensor)
                    assert encoded.data_ptr() == value.ctypes.data
                    np.testing.assert_array_equal(encoded.numpy(), value)
                else:
                    assert isinstance(encoded, np.ndarray)
    for name in ("transition_ids", "priorities"):
        if tensor_arrays:
            assert isinstance(state[name], torch.Tensor)
            assert state[name].data_ptr() == original_state[name].ctypes.data
    path = tmp_path / "replay.pt"
    torch.save(state, path)
    if tensor_arrays:
        # Arrays must be streamed as tensor storage, not embedded in pickle.
        with ZipFile(path) as archive:
            payload = archive.read("replay/data.pkl")
            assert b"numpy" not in payload
            assert any(name.startswith("replay/data/") for name in archive.namelist())
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    assert all(
        "initial_priority" not in trajectory
        for trajectory in restored.state_dict()["trajectories"]
    )

    for original, loaded in zip(
        replay.state_dict()["trajectories"],
        restored.state_dict()["trajectories"],
        strict=True,
    ):
        np.testing.assert_array_equal(
            loaded["initial_priorities"], original["initial_priorities"]
        )
        assert not loaded["initial_priorities"].flags.writeable

    assert len(restored) == len(replay)
    assert restored.trajectory_count == replay.trajectory_count
    np.testing.assert_array_equal(restored.priorities, replay.priorities)
    original_batch = replay.sample(4)
    restored_batch = restored.sample(4)
    for field in fields(original_batch):
        original = getattr(original_batch, field.name)
        loaded = getattr(restored_batch, field.name)
        if original is None:
            assert loaded is None
        else:
            torch.testing.assert_close(loaded, original)


@pytest.mark.parametrize("legacy_scalar", [False, True])
def test_replay_rejects_snapshot_without_initial_priorities(
    legacy_scalar: bool,
) -> None:
    replay = FIFOReplayBuffer(max_transitions=10)
    replay.add(make_trajectory(3, terminated=True))
    state = replay.state_dict()
    trajectory = state["trajectories"][0]
    del trajectory["initial_priorities"]
    if legacy_scalar:
        trajectory["initial_priority"] = 999.0

    restored = FIFOReplayBuffer(max_transitions=10)
    with pytest.raises(ValueError, match="replay trajectory fields are invalid"):
        restored.load_state_dict(state)
    assert len(restored) == 0


@pytest.mark.parametrize("truncated", [False, True])
def test_nonterminal_td_tails_use_available_rewards_and_zero_bootstrap(truncated) -> None:
    replay = FIFOReplayBuffer(10, unroll_steps=3, discount=0.5)
    replay.add(make_trajectory(
        4, truncated=truncated, predicted_values=(10.0, 11.0, 12.0, 13.0),
    ))
    stored = replay.state_dict()["trajectories"][0]
    np.testing.assert_allclose(stored["value_targets"], [3.125, 4.5, 5.0, 4.0, 0.0])
    np.testing.assert_array_equal(stored["value_valid_mask"], [True] * 5)
    np.testing.assert_allclose(
        stored["initial_priorities"],
        np.array([5.625, 6.5, 7.0, 9.0]) + replay.state_dict()["priority_epsilon"],
    )
    batch = replay.sample(4)
    for i, position in enumerate(batch.indices.tolist()):
        assert batch.value_mask[i, 0]
        assert bool(batch.value_bootstrap_mask[i, 0]) == (position == 0)
        if position > 0:
            assert batch.value_bootstrap_discounts[i, 0] == 0


def test_fifo_replay_evicts_oldest_complete_trajectories() -> None:
    replay = FIFOReplayBuffer(max_transitions=5, unroll_steps=2)

    replay.add(make_trajectory(3, block_id=0))
    result = replay.add(make_trajectory(3, block_id=1, initial_value=10))

    assert len(replay) == 3
    assert replay.trajectory_count == 1
    assert replay.utilization == pytest.approx(0.6)
    assert result.evicted_trajectories == 1
    assert result.evicted_transitions == 3

    batch = replay.sample(3)
    assert torch.all(batch.frames[:, 0] >= 10)
    assert batch.transition_ages is not None
    age_by_id = dict(zip(batch.indices.tolist(), batch.transition_ages.tolist()))
    assert age_by_id == {3: 2, 4: 1, 5: 0}
    final_sample = int((batch.frames[:, 0, 0, 0, 0] == 12).nonzero().item())
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
    assert batch.reanalysis_frames is not None
    assert batch.reanalysis_frames.shape == (3, 11, 1, 2, 2)
    shared_storage = batch.reanalysis_frames.untyped_storage().data_ptr()
    assert batch.frames.untyped_storage().data_ptr() == shared_storage
    assert batch.value_bootstrap_frames.untyped_storage().data_ptr() == shared_storage
    assert batch.value_bootstrap_values is not None
    assert batch.value_bootstrap_values.shape == (3, 6)
    assert batch.value_bootstrap_discounts is not None
    assert batch.value_bootstrap_discounts.shape == (3, 6)
    assert batch.value_bootstrap_mask is not None
    assert batch.value_bootstrap_mask.shape == (3, 6)
    assert batch.value_bootstrap_state_ids is not None
    assert batch.value_bootstrap_state_ids.shape == (3, 6)
    assert batch.value_bootstrap_state_ids.dtype == torch.long
    assert batch.indices.shape == (3,)
    assert batch.indices.dtype == torch.long
    assert batch.importance_weights.shape == (3,)
    assert batch.importance_weights.dtype == torch.float32
    assert batch.transition_ages is not None
    assert batch.transition_ages.shape == (3,)
    assert batch.transition_ages.dtype == torch.long
    assert batch.reanalysis_state_ids is not None
    assert batch.reanalysis_state_ids.shape == (3, 6)
    assert batch.reanalysis_state_ids.dtype == torch.long
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


def test_replay_can_skip_target_network_bootstrap_metadata() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=2,
        seed=3,
    )
    replay.add(make_trajectory(3, terminated=True))

    batch = replay.sample(
        batch_size=3,
        include_value_bootstraps=False,
    )

    assert batch.value_bootstrap_frames is None
    assert batch.reanalysis_frames is None
    assert batch.value_bootstrap_values is None
    assert batch.value_bootstrap_discounts is None
    assert batch.value_bootstrap_mask is None
    assert batch.value_bootstrap_state_ids is None


def test_reanalysis_state_ids_share_real_states_across_block_overlap() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=20,
        unroll_steps=2,
        seed=1,
    )
    replay.add(make_trajectory(7, block_id=0, lookahead_steps=2))
    replay.add(
        make_trajectory(
            3,
            block_id=1,
            initial_value=5,
            terminated=True,
        )
    )
    replay.add(
        make_trajectory(
            3,
            block_id=1,
            environment_index=1,
            initial_value=5,
            terminated=True,
        )
    )

    batch = replay.sample(batch_size=11)
    assert batch.reanalysis_state_ids is not None
    root_values = batch.frames[:, 0, 0, 0, 0]
    first_block_tail = int((root_values == 4).nonzero().item())
    # Lookahead states are no longer search roots of the preceding block.
    assert (batch.reanalysis_state_ids[first_block_tail, 1:] == -1).all()
    same_environment = int((batch.indices == 5).nonzero().item())
    other_environment = int((batch.indices == 8).nonzero().item())
    assert batch.reanalysis_state_ids[same_environment, 0] != batch.reanalysis_state_ids[other_environment, 0]
    preceding = int((root_values == 3).nonzero().item())
    assert batch.reanalysis_state_ids[preceding, 1] == batch.reanalysis_state_ids[first_block_tail, 0]
    assert batch.value_bootstrap_state_ids is not None
    first_root = int((root_values == 0).nonzero().item())
    first_bootstrap_root = int((root_values == 2).nonzero().item())
    assert (
        batch.value_bootstrap_state_ids[first_root, 0]
        == batch.reanalysis_state_ids[first_bootstrap_root, 0]
    )


def test_replay_builds_fixed_n_step_values_from_stored_root_values() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=3,
        discount=0.5,
        seed=3,
    )
    replay.add(make_trajectory(4, terminated=True))

    batch = replay.sample(batch_size=4)
    start_zero = int((batch.frames[:, 0, 0, 0, 0] == 0).nonzero().item())

    torch.testing.assert_close(
        batch.value_targets[start_zero],
        torch.tensor([3.125, 4.5, 5.0, 4.0]),
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
    assert batch.value_bootstrap_values[start_zero, 0] == 3.0
    assert batch.value_bootstrap_discounts[start_zero, 0] == 0.125
    assert batch.value_bootstrap_frames is not None
    assert batch.value_bootstrap_frames[start_zero, 0, 0, 0, 0] == 3


def test_replay_builds_value_targets_from_local_lookahead() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=6,
        unroll_steps=5,
        discount=0.5,
    )
    replay.add(make_trajectory(11, lookahead_steps=5))

    trajectory = replay._trajectories[0]
    expected = np.asarray(
        [
            sum(0.5**step * (offset + step + 1) for step in range(5))
            + (0.5**5 * (offset + 5) if offset + 5 < 6 else 0)
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


def test_replay_accepts_partial_lookahead_at_collection_cutoff() -> None:
    trajectory = make_trajectory(5, lookahead_steps=2)
    replay = FIFOReplayBuffer(max_transitions=3, unroll_steps=3)
    replay.add(trajectory)
    batch = replay.sample(batch_size=3)
    assert batch.value_mask[:, 0].all()
    assert not batch.value_bootstrap_mask.any()


def test_replay_stores_compact_overlapping_frame_context() -> None:
    replay = FIFOReplayBuffer(max_transitions=10, unroll_steps=1, seed=1)
    replay.add(make_trajectory(2, stack_size=4))

    batch = replay.sample(batch_size=2)

    assert batch.frames.shape == (2, 5, 1, 2, 2)
    first_sample = int((batch.frames[:, 0, 0, 0, 0] == 0).nonzero().item())
    torch.testing.assert_close(
        batch.frames[first_sample, :, 0, 0, 0],
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.uint8),
    )


def test_replay_unroll_stops_at_original_block_boundary() -> None:
    trajectory = make_trajectory(7, lookahead_steps=5)
    replay = FIFOReplayBuffer(max_transitions=10, unroll_steps=3, seed=2)
    replay.add(trajectory)

    batch = replay.sample(batch_size=2)
    crossing_sample = int((batch.frames[:, 0, 0, 0, 0] == 1).nonzero().item())

    torch.testing.assert_close(
        batch.frames[crossing_sample, :, 0, 0, 0],
        torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        batch.action_mask[crossing_sample],
        torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(
        batch.policy_mask[crossing_sample],
        torch.tensor([True, False, False, False]),
    )
    value_replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=3,
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
        torch.tensor([4.5, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        value_batch.value_mask[crossing_value_sample],
        torch.tensor([True, True, False, False]),
    )
    assert value_batch.value_bootstrap_values is not None
    assert value_batch.value_bootstrap_discounts is not None
    assert value_batch.value_bootstrap_mask is not None
    torch.testing.assert_close(
        value_batch.value_bootstrap_values[crossing_value_sample],
        torch.zeros(4),
    )
    torch.testing.assert_close(
        value_batch.value_bootstrap_discounts[crossing_value_sample],
        torch.zeros(4),
    )
    torch.testing.assert_close(
        value_batch.value_bootstrap_mask[crossing_value_sample],
        torch.zeros(4, dtype=torch.bool),
    )
    assert value_batch.value_bootstrap_frames is not None
    torch.testing.assert_close(
        value_batch.value_bootstrap_frames[crossing_value_sample, :, 0, 0, 0],
        torch.tensor([4, 5, 6, 7], dtype=torch.uint8),
    )


def reference_value_targets(
    trajectory: GameTrajectory,
    position: int,
    *,
    target_count: int,
    unroll_steps: int,
    discount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros(target_count, dtype=np.float32)
    valid = np.zeros(target_count, dtype=bool)
    bootstrap_mask = np.zeros(target_count, dtype=bool)
    stored_count = trajectory.stored_transition_count
    for offset in range(target_count):
        target_position = position + offset
        if target_position == len(trajectory):
            valid[offset] = True
            continue
        if target_position > len(trajectory):
            continue
        bootstrap_position = target_position + unroll_steps
        reward_end = min(bootstrap_position, stored_count)
        reward_return = sum(
            discount**step * trajectory.rewards[target_position + step]
            for step in range(reward_end - target_position)
        )
        if bootstrap_position < len(trajectory):
            reward_return += (
                discount**unroll_steps
                * trajectory.search_results[bootstrap_position].root_value
            )
            bootstrap_mask[offset] = True
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
        unroll_steps=3,
        discount=0.5,
        seed=5,
    )
    replay.add(trajectory)
    batch = replay.sample(len(trajectory))
    assert batch.value_bootstrap_mask is not None

    for batch_index in range(batch.batch_size):
        position = int(batch.frames[batch_index, 0, 0, 0, 0])
        expected_values, expected_valid, expected_bootstrap = reference_value_targets(
            trajectory,
            position,
            target_count=4,
            unroll_steps=3,
            discount=0.5,
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


@pytest.mark.parametrize("include_bootstraps", [False, True])
@pytest.mark.parametrize("block_length", [1, 4, 100])
@pytest.mark.parametrize("lookahead", [0, 1, 5])
@pytest.mark.parametrize("boundary", ["nonterminal", "terminated", "truncated"])
def test_v2_atari_block_boundaries_match_reference(
    include_bootstraps, block_length, lookahead, boundary,
) -> None:
    horizon = 5
    trajectory = make_trajectory(
        block_length + lookahead, lookahead_steps=lookahead,
        terminated=boundary == "terminated", truncated=boundary == "truncated",
    )
    replay = FIFOReplayBuffer(block_length, unroll_steps=horizon, discount=0.5)
    replay.add(trajectory)
    batch = replay.sample(block_length, include_value_bootstraps=include_bootstraps)
    for row, raw_start in enumerate(batch.indices):
        start = int(raw_start)
        real_actions = min(horizon, block_length - start)
        np.testing.assert_array_equal(
            batch.action_mask[row], np.arange(horizon) < real_actions
        )
        np.testing.assert_array_equal(
            batch.value_mask[row], np.arange(horizon + 1) <= real_actions
        )
        np.testing.assert_array_equal(
            batch.policy_mask[row], start + np.arange(horizon + 1) < block_length
        )
        # V2 masks by the action just taken: the boundary successor has zero
        # policy/value targets but still a real reward/consistency/value loss.
        expected, _, bootstrap_mask = reference_value_targets(
            trajectory, start, target_count=horizon + 1,
            unroll_steps=horizon, discount=0.5,
        )
        np.testing.assert_allclose(batch.value_targets[row], expected)
        np.testing.assert_array_equal(
            batch.actions[row, :real_actions, 0], trajectory.actions[start:start + real_actions]
        )
        assert ((batch.actions[row] >= 0) & (batch.actions[row] < 2)).all()
        assert not batch.rewards[row, real_actions:].any()
        if include_bootstraps:
            np.testing.assert_array_equal(batch.value_bootstrap_mask[row], bootstrap_mask)
        # Frame context is retained independently of action validity.
        for offset in range(horizon + 1):
            assert batch.frames[row, offset, 0, 0, 0] == min(
                start + offset, trajectory.stored_transition_count
            )


def test_prioritized_replay_matches_efficientzero_v1_atari() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=1,
        seed=4,
    )
    replay.add(make_trajectory(3, terminated=True))
    replay.update_priorities(np.array([0, 1, 2]), np.array([1.0, 4.0, 16.0]))

    batch = replay.sample(batch_size=3)

    probabilities = np.array([1.0, 4.0, 16.0]) ** 0.6
    probabilities /= probabilities.sum()
    expected_weights = (3 * probabilities[batch.indices.numpy()]) ** -0.4
    expected_weights /= expected_weights.max()
    np.testing.assert_allclose(
        batch.importance_weights.numpy(), expected_weights, rtol=1e-6
    )
    assert batch.importance_weights.min() > 0.1

    counts = np.zeros(3, dtype=np.int64)
    for _ in range(300):
        sampled = replay.sample(1)
        counts[int(sampled.indices.item())] += 1
    assert counts[2] > counts[1] > counts[0]


@pytest.mark.parametrize("clip", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.6, 0.4)])
def test_prioritized_replay_clips_weights(clip, alpha, beta) -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=1,
        priority_weight_clip=clip,
        priority_alpha=alpha,
        priority_beta=0.2,
        seed=4,
    )
    replay.add(make_trajectory(3, terminated=True))
    replay.update_priorities(np.array([0, 1, 2]), np.array([1.0, 4.0, 16.0]))

    # Clipping preserves configured alpha and the per-sample beta override.
    batch = replay.sample(batch_size=3, priority_beta=beta)
    probabilities = np.array([1.0, 4.0, 16.0]) ** alpha
    probabilities /= probabilities.sum()
    expected_weights = (3 * probabilities[batch.indices.numpy()]) ** -beta
    expected_weights /= expected_weights.max()
    expected_weights = expected_weights.clip(clip, 1.0)
    np.testing.assert_allclose(
        batch.importance_weights.numpy(), expected_weights, rtol=1e-6
    )
    assert batch.importance_weights.min() >= clip


def test_default_weight_clip_does_not_floor_weights() -> None:
    replay = FIFOReplayBuffer(10, priority_alpha=1.0, priority_beta=1.0)
    replay.add(make_trajectory(3, terminated=True))
    replay.update_priorities([0, 1, 2], [1.0, 4.0, 100.0])

    batch = replay.sample(3)

    assert batch.importance_weights.min() == pytest.approx(0.01)
    assert batch.importance_weights.max() == pytest.approx(1.0)


def test_new_trajectory_error_can_exceed_current_maximum_priority() -> None:
    replay = FIFOReplayBuffer(max_transitions=10, priority_epsilon=1e-6)
    replay.add(make_trajectory(2, terminated=True))
    replay.update_priorities([0, 1], [2.0, 5.0])

    replay.add(
        make_trajectory(
            1,
            episode_id=1,
            initial_value=10,
            terminated=True,
            predicted_values=(10.0,),
        )
    )

    np.testing.assert_allclose(replay.priorities, [2.0, 5.0, 9.000001])


@pytest.mark.parametrize("priority_weight_clip", [0.0, 0.1])
def test_insertion_uses_v2_maximum_but_preserves_individual_errors(priority_weight_clip: float) -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=1,
        discount=0.5,
        priority_weight_clip=priority_weight_clip,
        priority_epsilon=1e-6,
    )
    replay.add(
        make_trajectory(
            3,
            lookahead_steps=1,
            predicted_values=(10.0, 20.0, 30.0),
        )
    )

    # Prediction-bootstrapped targets are [11, 17], giving errors [1, 3].
    # Lookahead is not inserted; both starts receive the trajectory maximum.
    np.testing.assert_allclose(replay.priorities, [3.000001, 3.000001])

    replay.update_priorities([0, 1], [100.0, 200.0])
    initial = replay.state_dict()["trajectories"][0]["initial_priorities"]
    np.testing.assert_allclose(initial, [1.000001, 3.000001])
    assert not initial.flags.writeable
    replay.add(
        make_trajectory(
            3,
            episode_id=1,
            lookahead_steps=1,
            predicted_values=(10.0, 20.0, 30.0),
        )
    )
    np.testing.assert_allclose(
        replay.priorities, [100.0, 200.0, 200.0, 200.0]
    )


@pytest.mark.parametrize("priority_weight_clip", [0.0, 0.1])
def test_initial_priorities_handle_terminal_tail_and_zero_error(priority_weight_clip: float) -> None:
    replay = FIFOReplayBuffer(
        max_transitions=10,
        unroll_steps=2,
        discount=0.5,
        priority_weight_clip=priority_weight_clip,
        priority_epsilon=1e-6,
    )
    replay.add(
        make_trajectory(
            3, terminated=True, predicted_values=(2.75, 4.0, 3.0)
        )
    )
    # Targets: [1 + .5*2 + .25*3, 2 + .5*3, 3]. Terminal-tail
    # returns do not bootstrap, and exact predictions retain positive epsilon.
    initial = replay.state_dict()["trajectories"][0]["initial_priorities"]
    np.testing.assert_allclose(initial, [1e-6, 0.500001, 1e-6])
    # Empty-buffer maximum is 1, not the maximum error (0.500001).
    np.testing.assert_array_equal(replay.priorities, [1.0, 1.0, 1.0])

    restored = FIFOReplayBuffer(
        max_transitions=10, unroll_steps=2,
        discount=0.5, priority_weight_clip=priority_weight_clip, priority_epsilon=1e-6,
    )
    restored.load_state_dict(replay.state_dict())
    np.testing.assert_array_equal(restored.priorities, replay.priorities)


def test_first_replay_transitions_use_maximum_error_above_one() -> None:
    replay = FIFOReplayBuffer(
        max_transitions=2,
        unroll_steps=1,
        discount=0.5,
        priority_epsilon=1e-6,
    )
    replay.add(
        make_trajectory(
            3,
            lookahead_steps=1,
            predicted_values=(10.0, 20.0, 30.0),
        )
    )

    np.testing.assert_allclose(replay.priorities, [3.000001, 3.000001])


@pytest.mark.parametrize("priority_weight_clip", [0.0, 0.1])
@pytest.mark.parametrize("capacity", [3, 6])
@pytest.mark.parametrize(
    "existing,errors,expected",
    [
        (None, [0.0, 0.0, 0.0], 1.0),
        (None, [0.0, 0.5, 2.0], 2.000001),
        ([2.0, 10.0, 3.0], [0.01, 0.5, 2.0], 10.0),
        ([2.0, 10.0, 3.0], [0.01, 20.0, 2.0], 20.000001),
        # Neither a permanent floor of 1 nor a historical maximum is used.
        ([0.1, 0.2, 0.3], [0.0, 0.0, 0.0], 0.3),
        ([0.1, 0.2, 0.3], [0.0, 0.5, 0.0], 0.500001),
    ],
)
@pytest.mark.parametrize("use_max_priority", [False, True])
def test_maximum_insertion_branches(priority_weight_clip, capacity, existing, errors, expected, use_max_priority):
    replay = FIFOReplayBuffer(
        max_transitions=capacity, unroll_steps=1,
        discount=0.0, priority_weight_clip=priority_weight_clip, priority_epsilon=1e-6,
        use_max_priority=use_max_priority,
    )
    if use_max_priority:
        expected = max(existing) if existing is not None else 1.0
    if existing is not None:
        # Start with high errors, then lower them through learner updates.
        replay.add(make_trajectory(3, terminated=True, predicted_values=(101., 102., 103.)))
        replay.update_priorities([0, 1, 2], existing)
    replay.add(make_trajectory(
        3, episode_id=1, terminated=True,
        predicted_values=tuple(i + 1 + error for i, error in enumerate(errors)),
    ))
    # At capacity=3 the previous maximum is evicted by this insertion, but
    # must still seed the new trajectory, matching insertion-before-eviction.
    prefix = existing if existing is not None and capacity == 6 else []
    np.testing.assert_allclose(replay.priorities, [*prefix, *([expected] * 3)])
    # Sampling-mode differences must not prevent individual priority updates.
    first_id = 3 if existing is not None else 0
    replay.update_priorities([first_id], [0.125])
    assert replay.priorities[-3] == 0.125
    np.testing.assert_allclose(replay.priorities[-2:], [expected, expected])


@pytest.mark.parametrize("priority_weight_clip", [0.0, 0.1])
@pytest.mark.parametrize("use_max_priority", [False, True])
def test_maximum_insertion_extends_sequentially_and_survives_restore(priority_weight_clip, use_max_priority):
    replay = FIFOReplayBuffer(
        max_transitions=10, unroll_steps=1, discount=0.0,
        priority_weight_clip=priority_weight_clip, use_max_priority=use_max_priority,
    )
    replay.extend([
        make_trajectory(2, episode_id=0, terminated=True, predicted_values=(1., 22.)),
        make_trajectory(2, episode_id=1, terminated=True, predicted_values=(1., 2.)),
    ])
    np.testing.assert_allclose(
        replay.priorities, [1.0 if use_max_priority else 20.000001] * 4
    )
    # Restore current priorities verbatim, not original insertion errors.
    replay.update_priorities(range(4), [0.1, 0.2, 0.3, 0.4])
    restored = FIFOReplayBuffer(
        max_transitions=10, unroll_steps=1, discount=0.0,
        priority_weight_clip=priority_weight_clip, use_max_priority=use_max_priority,
    )
    restored.load_state_dict(replay.state_dict())
    np.testing.assert_array_equal(restored.priorities, replay.priorities)
    restored.add(make_trajectory(2, episode_id=2, terminated=True, predicted_values=(1., 2.)))
    np.testing.assert_allclose(restored.priorities, [0.1, 0.2, 0.3, 0.4, 0.4, 0.4])


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
        ("unroll_steps", 1.5, TypeError),
        ("discount", -0.1, ValueError),
        ("discount", 1.1, ValueError),
        ("discount", float("nan"), ValueError),
        ("use_max_priority", 1, TypeError),
        ("use_max_priority", "false", TypeError),
        ("priority_weight_clip", -0.1, ValueError),
        ("priority_weight_clip", 1.1, ValueError),
        ("priority_weight_clip", float("nan"), ValueError),
        ("priority_weight_clip", float("inf"), ValueError),
        ("priority_alpha", -0.1, ValueError),
        ("priority_alpha", 1.1, ValueError),
        ("priority_beta", -0.1, ValueError),
        ("priority_beta", 1.1, ValueError),
        ("priority_epsilon", 0.0, ValueError),
        ("priority_epsilon", float("nan"), ValueError),
    ],
)
def test_replay_validates_fixed_target_configuration(
    keyword: str, value: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        FIFOReplayBuffer(3, **{keyword: value})  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_replay_can_construct_batches_in_pinned_memory() -> None:
    replay = FIFOReplayBuffer(3, unroll_steps=1)
    replay.add(make_trajectory(2, terminated=True))

    batch = replay.sample(2, pin_memory=True)

    assert batch.frames.is_pinned()
    assert batch.reanalysis_frames is not None
    assert batch.reanalysis_frames.is_pinned()
    assert batch.value_bootstrap_frames is not None
    assert batch.value_bootstrap_frames.is_pinned()
    assert (
        batch.frames.untyped_storage().data_ptr()
        == batch.reanalysis_frames.untyped_storage().data_ptr()
    )
    assert batch.actions.is_pinned()
    assert batch.value_targets.is_pinned()
    assert batch.indices.is_pinned()


def test_replay_sample_does_not_accept_target_configuration_overrides() -> None:
    replay = FIFOReplayBuffer(3, unroll_steps=1, discount=0.5)
    replay.add(make_trajectory(1, terminated=True))

    assert replay.unroll_steps == 1
    assert replay.discount == 0.5
    with pytest.raises(AttributeError):
        replay.unroll_steps = 2  # type: ignore[misc]
    with pytest.raises(TypeError, match="unexpected keyword"):
        replay.sample(1, unroll_steps=2)  # type: ignore[call-arg]


def test_replay_prepares_read_only_contiguous_targets_on_add() -> None:
    replay = FIFOReplayBuffer(3, unroll_steps=2, discount=0.5)
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
    replay = FIFOReplayBuffer(2, unroll_steps=1, seed=1)
    replay.add(make_trajectory(2, terminated=True))
    invalid = make_trajectory(
        2,
        episode_id=1,
        initial_value=10,
        terminated=True,
    )
    incompatible_result = SearchResult(
        action=0,
        policy_target=(1.0, 0.0, 0.0),
        root_value=0.0,
    )
    invalid = replace(
        invalid,
        search_results=(incompatible_result, *invalid.search_results[1:]),
    )

    with pytest.raises(ValueError, match="same size"):
        replay.add(invalid)

    assert len(replay) == 2
    assert replay.trajectory_count == 1
    batch = replay.sample(2)
    assert sorted(batch.frames[:, 0, 0, 0, 0].tolist()) == [0, 1]


def test_timeout_trains_available_returns_with_zero_bootstrap() -> None:
    replay = FIFOReplayBuffer(
        10, unroll_steps=3, discount=0.5,
    )
    trajectory = make_trajectory(4, truncated=True)
    replay.add(trajectory)
    stored = replay.state_dict()["trajectories"][0]
    assert stored["truncated"] and not stored["terminated"]
    np.testing.assert_array_equal(
        stored["value_valid_mask"], [True, True, True, True, True]
    )
    batch = replay.sample(4)
    for i in range(batch.batch_size):
        position = int(batch.frames[i, 0, 0, 0, 0])
        values, valid, bootstrap = reference_value_targets(
            trajectory, position,
            target_count=4, unroll_steps=3, discount=0.5,
        )
        np.testing.assert_allclose(batch.value_targets[i], values)
        np.testing.assert_array_equal(batch.value_mask[i], valid)
        np.testing.assert_array_equal(batch.value_bootstrap_mask[i], bootstrap)


@pytest.mark.parametrize("length", [1, 4, 8])
def test_v1_timeout_targets_match_terminal_targets(length: int) -> None:
    options = dict(max_transitions=20, unroll_steps=3, discount=0.5)
    replay = FIFOReplayBuffer(**options, treat_truncations_as_terminal=True)
    terminal_replay = FIFOReplayBuffer(**options)
    replay.add(make_trajectory(length, truncated=True))
    terminal_replay.add(make_trajectory(length, terminated=True))
    stored = replay.state_dict()["trajectories"][0]
    expected = terminal_replay.state_dict()["trajectories"][0]
    assert stored["truncated"] and not stored["terminated"]
    for name in ("value_targets", "value_valid_mask", "initial_priorities", "rewards"):
        np.testing.assert_array_equal(stored[name], expected[name])
    assert stored["value_targets"][-1] == 0
    assert stored["value_targets"][-2] == length
    assert stored["value_valid_mask"].all()
    actual_batch = replay.sample(length)
    expected_batch = terminal_replay.sample(length)
    for name in (
        "value_targets", "value_mask", "value_bootstrap_mask",
        "value_bootstrap_values", "rewards", "policy_mask", "action_mask",
    ):
        torch.testing.assert_close(getattr(actual_batch, name), getattr(expected_batch, name))


@pytest.mark.parametrize("enabled", [False, True])
def test_timeout_target_mode_checkpoint_compatibility(enabled: bool) -> None:
    replay = FIFOReplayBuffer(10, treat_truncations_as_terminal=enabled)
    replay.add(make_trajectory(4, truncated=True))
    state = replay.state_dict()
    restored = FIFOReplayBuffer(10, treat_truncations_as_terminal=enabled)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(
        restored.state_dict()["trajectories"][0]["value_valid_mask"],
        state["trajectories"][0]["value_valid_mask"],
    )
    with pytest.raises(ValueError, match="treat_truncations_as_terminal"):
        FIFOReplayBuffer(10, treat_truncations_as_terminal=not enabled).load_state_dict(state)
    if not enabled:
        del state["treat_truncations_as_terminal"]
        restored.load_state_dict(state)
        with pytest.raises(ValueError, match="treat_truncations_as_terminal"):
            FIFOReplayBuffer(10, treat_truncations_as_terminal=True).load_state_dict(state)


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_timeout_target_mode_requires_boolean(invalid) -> None:
    with pytest.raises(TypeError, match="treat_truncations_as_terminal"):
        FIFOReplayBuffer(10, treat_truncations_as_terminal=invalid)


def test_truncated_targets_survive_checkpoint_round_trip() -> None:
    replay = FIFOReplayBuffer(10)
    replay.add(make_trajectory(4, truncated=True))
    restored = FIFOReplayBuffer(10)
    restored.load_state_dict(replay.state_dict())
    np.testing.assert_allclose(restored.priorities, replay.priorities)
    expected = replay.state_dict()["trajectories"][0]
    actual = restored.state_dict()["trajectories"][0]
    assert actual["truncated"] and not actual["terminated"]
    np.testing.assert_allclose(actual["value_targets"], expected["value_targets"])
    np.testing.assert_array_equal(
        actual["value_valid_mask"], expected["value_valid_mask"]
    )
