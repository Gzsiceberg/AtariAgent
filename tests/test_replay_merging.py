"""Temporary terminals disappear when their continuation owners arrive."""

from dataclasses import replace
import weakref

import numpy as np
import pytest
import torch

from atariagent.replay import FIFOReplayBuffer
from atariagent.search import SearchResult
from atariagent.selfplay import GameTrajectory


def block(start, count, *, block_id=0, lookahead=0, environment=0, episode=0,
          terminated=False, truncated=False):
    stored = count + lookahead
    return GameTrajectory(
        environment_index=environment, episode_id=episode, block_id=block_id,
        stack_size=2,
        frames=tuple(np.full((1, 2, 2), i % 256, dtype=np.uint8)
                     for i in range(start, start + stored + 2)),
        actions=tuple(i % 2 for i in range(start, start + stored)),
        rewards=(1.0,) * stored, raw_rewards=(1.0,) * stored,
        search_results=tuple(SearchResult(
            action=i % 2, policy_target=(0.25, 0.75), root_value=10.0,
        ) for i in range(start, start + stored)),
        predicted_values=(10.0,) * stored,
        terminated=terminated, truncated=truncated,
        full_episode_done=(terminated or truncated) and not lookahead,
        lookahead_steps=lookahead,
    )


def row(batch, transition_id):
    return int((batch.indices == transition_id).nonzero().item())


@pytest.mark.parametrize("include_bootstraps", [False, True])
def test_405_becomes_805_with_interleaved_environment_and_stable_ids(include_bootstraps):
    replay = FIFOReplayBuffer(1000, unroll_steps=5, discount=0.5)
    replay.add(block(0, 400, lookahead=5))
    before = replay.sample(400, include_value_bootstraps=include_bootstraps)
    old = row(before, 399)
    assert before.value_targets[old].tolist() == [1, 0, 0, 0, 0, 0]
    assert before.action_mask[old].tolist() == [True, False, False, False, False]
    assert before.policy_mask[old].tolist() == [True, False, False, False, False, False]
    assert before.terminal_mask[old].tolist() == [False, True, False, False, False, False]
    assert before.value_mask[old].tolist() == [True, True, False, False, False, False]
    # All post-boundary frames are padded at the temporary endpoint.
    assert before.frames[old, 2:, 0, 0, 0].tolist() == [401 % 256] * 5
    replay.update_priorities([399], [7.0])

    replay.add(block(0, 2, environment=1))
    replay.add(block(400, 400, block_id=1, lookahead=5))
    assert len(replay) == 802
    merged = replay._trajectories[0]
    ids = merged.transition_ids
    offset = replay._trajectory_offsets[402]
    assert replay.trajectory_count == 2
    assert len(merged) == 800 and merged.stored_transition_count == 805
    assert offset == 400
    assert merged.frames.shape[0] == 807
    assert merged.rewards.tolist() == [1.0] * 805
    assert merged.actions.tolist() == [i % 2 for i in range(805)]
    assert replay.priorities[399] == 7.0
    assert replay._trajectories[1] is not merged

    after = replay.sample(802, include_value_bootstraps=include_bootstraps)
    old_after, new_root = row(after, 399), row(after, 402)
    assert after.value_targets[old_after].tolist() == [2.25] * 6
    assert after.action_mask[old_after].all()
    assert after.policy_mask[old_after].all()
    assert not after.terminal_mask[old_after].any()
    assert after.reanalysis_state_ids[old_after, 0] == before.reanalysis_state_ids[old, 0]
    assert after.reanalysis_state_ids[old_after, 1] == after.reanalysis_state_ids[new_root, 0]
    if include_bootstraps:
        assert after.value_bootstrap_mask[old_after].all()
        assert after.value_bootstrap_state_ids[old_after, 0] == ids[404]
    last = row(after, 801)
    assert after.terminal_mask[last, 1]
    assert after.value_targets[last, 0] == 1
    # Merging is immutable: a previously sampled batch remains unchanged.
    assert before.terminal_mask[old, 1]
    assert before.value_targets[old, 0] == 1
    replay.update_priorities([399, 402, 801], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(replay.priorities[[399, 402, 801]], [3, 4, 5])


@pytest.mark.parametrize("ending", ["continuing", "terminated", "truncated", "flush"])
@pytest.mark.parametrize("restore_before", [False, True])
def test_short_continuation_merges_and_restores(ending, restore_before):
    options = dict(max_transitions=20, unroll_steps=3, discount=0.5)
    replay = FIFOReplayBuffer(**options)
    replay.add(block(0, 4, lookahead=2))
    if restore_before:
        restored = FIFOReplayBuffer(**options)
        restored.load_state_dict(replay.state_dict())
        replay = restored
    replay.add(block(
        4, 3, block_id=1, lookahead=2 if ending == "continuing" else 0,
        terminated=ending == "terminated", truncated=ending == "truncated",
    ))
    merged = replay._trajectories[0]
    assert replay.trajectory_count == 1
    assert len(merged) == 7
    assert merged.stored_transition_count == (9 if ending == "continuing" else 7)
    batch = replay.sample(7)
    crossing = row(batch, 3)
    assert batch.value_targets[crossing, 0] == 3.0  # 1 + .5 + .25 + .125*10
    assert batch.policy_mask[crossing].all()
    assert not batch.terminal_mask[crossing].any()
    last = row(batch, 6)
    assert batch.value_targets[last].tolist() == [1, 0, 0, 0]
    assert batch.value_mask[last].tolist() == [True, True, False, False]
    assert batch.policy_mask[last].tolist() == [True, False, False, False]
    restored = FIFOReplayBuffer(**options)
    restored.load_state_dict(replay.state_dict(tensor_arrays=True))
    expected, actual = replay.sample(7), restored.sample(7)
    for name in ("indices", "frames", "value_targets", "terminal_mask",
                 "reachable_mask", "policy_targets", "value_bootstrap_mask"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    assert restored._trajectory_offsets[4] == 4
    assert restored.trajectory_count == 1


@pytest.mark.parametrize("reason", ["terminated", "truncated", "episode", "gap", "reset_frames"])
def test_no_merge_across_actual_boundaries_or_missing_blocks(reason):
    replay = FIFOReplayBuffer(20, unroll_steps=3)
    replay.add(block(0, 4, terminated=reason == "terminated", truncated=reason == "truncated"))
    continuation = block(
        4 if reason != "reset_frames" else 20, 4,
        block_id=2 if reason == "gap" else 1,
        episode=1 if reason == "episode" else 0,
    )
    replay.add(continuation)
    assert replay.trajectory_count == 2
    batch = replay.sample(8)
    assert batch.terminal_mask[row(batch, 3), 1]
    assert not batch.policy_mask[row(batch, 3), 1]


def test_terminal_in_older_lookahead_does_not_prevent_merge():
    replay = FIFOReplayBuffer(20, unroll_steps=3)
    replay.add(block(0, 4, lookahead=2, terminated=True))
    replay.add(block(4, 2, block_id=1, terminated=True))
    assert replay.trajectory_count == 1
    merged = replay._trajectories[0]
    assert len(merged) == merged.stored_transition_count == 6
    batch = replay.sample(6)
    crossing = row(batch, 3)
    assert batch.policy_mask[crossing].tolist() == [True, True, True, False]
    assert batch.terminal_mask[crossing].tolist() == [False, False, False, True]


def test_replay_grows_past_budget_without_eviction_or_id_changes():
    replay = FIFOReplayBuffer(8, unroll_steps=3)
    replay.add(block(0, 4, lookahead=2))
    replay.add(block(4, 4, block_id=1, lookahead=2))
    previous = replay.sample(8)
    previous_id = previous.reanalysis_state_ids[row(previous, 4), 0].item()
    result = replay.add(block(8, 4, block_id=2, terminated=True))
    assert result.evicted_transitions == 0 and result.added_transitions == 4
    assert len(replay) == 12
    assert replay.trajectory_count == 1
    current = replay.sample(12)
    assert sorted(current.indices.tolist()) == list(range(12))
    assert current.reanalysis_state_ids[row(current, 4), 0] == previous_id
    assert current.policy_mask[row(current, 7)].all()
    assert not current.terminal_mask[row(current, 7)].any()
    replay.update_priorities([0, 4, 11], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(replay.priorities[[0, 4, 11]], [1, 2, 3])
    restored = FIFOReplayBuffer(8, unroll_steps=3)
    restored.load_state_dict(replay.state_dict())
    assert len(restored) == 12 and restored.trajectory_count == 1


def test_multiple_short_blocks_and_oversized_lookahead_merge_without_duplicates():
    replay = FIFOReplayBuffer(20, unroll_steps=5)
    for i in range(4):
        replay.add(block(i, 1, block_id=i, lookahead=5))
    merged = replay._trajectories[0]
    assert replay.trajectory_count == 1
    assert len(merged) == 4 and merged.stored_transition_count == 9
    batch = replay.sample(4)
    assert batch.policy_mask[row(batch, 0)].tolist() == [True] * 4 + [False] * 2
    assert batch.terminal_mask[row(batch, 0)].tolist() == [False] * 4 + [True, False]


def test_arriving_owner_replaces_inactive_lookahead_targets():
    replay = FIFOReplayBuffer(20, unroll_steps=2, discount=0.5)
    first = block(0, 4, lookahead=2)
    # Stored lookahead must never leak into the temporary-terminal return.
    first = replace(first, rewards=(1, 1, 1, 1, 999, 999))
    replay.add(first)
    before = replay.sample(4)
    assert before.value_targets[row(before, 3), 0] == 1
    replay.add(block(4, 4, block_id=1))
    after = replay.sample(8)
    assert after.value_targets[row(after, 3), 0] == 4.0
    assert after.rewards[row(after, 3)].tolist() == [1, 1]


@pytest.mark.parametrize("mode", ["td", "search", "mixed"])
def test_reused_batch_and_target_selection_remove_old_terminal(mode):
    replay = FIFOReplayBuffer(20, unroll_steps=2, discount=0.5)
    replay.add(block(0, 4, lookahead=2))
    arrays = replay._allocate_batch_arrays(1, include_value_bootstraps=True)

    def fill():
        replay._fill_batch_arrays(
            arrays, replay._locations_for_indices(np.array([3])),
            transition_ids=np.array([3]), importance_weights=np.ones(1),
        )
        batch = replay._batch_from_arrays(arrays)
        batch = batch.with_reanalysis_targets(
            value_targets=batch.value_targets,
            policy_targets=batch.policy_targets,
            search_value_targets=torch.full_like(batch.value_targets, 8),
        )
        return batch.with_selected_value_targets(
            mode=mode, learner_step=10, collection_steps=10,
            mixed_start_step=0, freshness_threshold=0,
        )

    before = fill()
    assert before.terminal_mask[0, 1]
    assert before.value_targets[0, 1] == 0
    assert not before.value_bootstrap_mask.any()
    replay.add(block(4, 4, block_id=1))
    after = fill()
    assert not after.terminal_mask.any()
    assert after.policy_mask.all() and after.action_mask.all()
    assert after.value_bootstrap_mask.all()
    assert after.value_targets[0, 1] == (4 if mode == "td" else 8)


def test_concat_releases_original_chunk_arrays():
    replay = FIFOReplayBuffer(4, unroll_steps=2)
    replay.add(block(0, 4, lookahead=2))
    previous_frames = weakref.ref(replay._trajectories[0].frames)
    previous_rewards = weakref.ref(replay._trajectories[0].rewards)
    replay.add(block(4, 4, block_id=1, lookahead=2))
    assert previous_frames() is None and previous_rewards() is None
    assert replay.trajectory_count == 1
    episode = replay._trajectories[0]
    for name in ("frames", "actions", "rewards", "policy_targets", "root_values",
                 "predicted_values", "transition_ids", "value_targets"):
        array = getattr(episode, name)
        assert array.flags.c_contiguous and not array.flags.writeable
    assert episode.stored_transition_count == 10
    assert len(replay.state_dict()["trajectories"]) == 1


def test_sampling_location_lookup_uses_only_precomputed_tables(monkeypatch):
    replay = FIFOReplayBuffer(20, unroll_steps=2)
    replay.add(block(0, 4, lookahead=2))
    replay.add(block(0, 3, environment=1))
    replay.add(block(4, 4, block_id=1, terminated=True))

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling must not rebuild episode/block indexes")

    for operation in ("concatenate", "cumsum", "searchsorted", "fromiter"):
        monkeypatch.setattr(np, operation, forbidden)
    slots = np.array([0, 4, 7, 3, 10])
    locations = replay._locations_for_indices(slots)
    assert [offset for _, _, offset in locations] == [0, 0, 4, 3, 7]
    assert [episode.environment_index for episode, _, _ in locations] == [0, 1, 0, 0, 0]
    for slot, (_, ids, offset) in zip(slots, locations, strict=True):
        assert ids[offset] == slot


@pytest.mark.parametrize("tensor_arrays", [False, True])
def test_interleaved_episode_snapshot_restores_ids_maps_and_can_continue(tensor_arrays):
    options = dict(max_transitions=4, unroll_steps=2)
    replay = FIFOReplayBuffer(**options)
    replay.add(block(0, 4, lookahead=2))
    replay.add(block(0, 3, environment=1))
    replay.add(block(4, 4, block_id=1, lookahead=2))
    state = replay.state_dict(tensor_arrays=tensor_arrays)
    assert len(state["trajectories"]) == 2
    restored = FIFOReplayBuffer(**options)
    restored.load_state_dict(state)
    for candidate in (replay, restored):
        candidate.add(block(3, 3, block_id=1, environment=1, terminated=True))
        candidate.add(block(8, 2, block_id=2, terminated=True))
        assert candidate.trajectory_count == 2
        assert len(candidate) == 16
        with pytest.raises(ValueError, match="identity already exists"):
            candidate.add(block(4, 4, block_id=1))
    expected, actual = replay.sample(16), restored.sample(16)
    for name in ("indices", "frames", "actions", "value_targets", "policy_targets",
                 "reanalysis_state_ids", "value_bootstrap_state_ids", "transition_ages"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(restored._trajectory_rows, replay._trajectory_rows)
    np.testing.assert_array_equal(restored._trajectory_offsets, replay._trajectory_offsets)


@pytest.mark.parametrize("corruption", ["duplicate", "out_of_range", "missing"])
def test_snapshot_rejects_invalid_episode_transition_ids_without_mutation(corruption):
    replay = FIFOReplayBuffer(10, unroll_steps=2)
    replay.add(block(0, 4))
    state = replay.state_dict()
    ids = state["trajectories"][0]["transition_ids"].copy()
    if corruption == "duplicate":
        ids[1] = ids[0]
    elif corruption == "out_of_range":
        ids[1] = 10
    else:
        ids = ids[:-1]
    state["trajectories"][0]["transition_ids"] = ids
    restored = FIFOReplayBuffer(10, unroll_steps=2)
    with pytest.raises(ValueError, match="episode transition IDs"):
        restored.load_state_dict(state)
    assert len(restored) == 0
