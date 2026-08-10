import numpy as np
import pytest
import torch

from atariagent.replay import FIFOReplayBuffer
from atariagent.search import Node, SearchResult
from atariagent.selfplay import GameTrajectory


def make_trajectory(
    length: int,
    *,
    block_id: int = 0,
    episode_id: int = 0,
    initial_value: int = 0,
    stack_size: int = 1,
    terminated: bool = False,
) -> GameTrajectory:
    frames = tuple(
        np.full((1, 2, 2), initial_value + step, dtype=np.uint8)
        for step in range(length + stack_size)
    )
    results = tuple(
        SearchResult(
            action=step % 2,
            policy=(0.25, 0.75),
            visit_counts=(1, 3),
            root_value=float(initial_value + step),
            root=Node(prior=1.0),
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
        truncated=False,
    )


def test_fifo_replay_evicts_oldest_complete_trajectories() -> None:
    replay = FIFOReplayBuffer(max_transitions=5)

    replay.add(make_trajectory(3, block_id=0))
    result = replay.add(make_trajectory(3, block_id=1, initial_value=10))

    assert len(replay) == 3
    assert replay.trajectory_count == 1
    assert replay.utilization == pytest.approx(0.6)
    assert result.evicted_trajectories == 1
    assert result.evicted_transitions == 3

    batch = replay.sample(3, unroll_steps=2)
    assert torch.all(batch.frames[:, 0] >= 10)
    final_sample = int(
        (batch.frames[:, 0, 0, 0, 0] == 12).nonzero().item()
    )
    torch.testing.assert_close(
        batch.target_mask[final_sample],
        torch.tensor([True, False, False]),
    )


def test_replay_samples_padded_five_step_tensor_batches() -> None:
    replay = FIFOReplayBuffer(max_transitions=10, seed=3)
    replay.add(make_trajectory(3, terminated=True))

    batch = replay.sample(batch_size=3, unroll_steps=5)

    assert batch.frames.shape == (3, 6, 1, 2, 2)
    assert batch.frames.dtype == torch.uint8
    assert batch.actions.shape == (3, 5, 1)
    assert batch.actions.dtype == torch.long
    assert batch.rewards.shape == (3, 5)
    assert batch.policy_targets.shape == (3, 6, 2)
    assert batch.root_values.shape == (3, 6)
    assert batch.value_targets.shape == (3, 6)
    assert batch.action_mask.dtype == torch.bool
    assert batch.target_mask.dtype == torch.bool
    assert batch.value_mask.dtype == torch.bool

    sampled_start_values = batch.frames[:, 0, 0, 0, 0]
    assert sorted(sampled_start_values.tolist()) == [0, 1, 2]
    final_sample = int((sampled_start_values == 2).nonzero().item())
    torch.testing.assert_close(
        batch.action_mask[final_sample],
        torch.tensor([True, False, False, False, False]),
    )
    torch.testing.assert_close(
        batch.target_mask[final_sample],
        torch.tensor([True, True, False, False, False, False]),
    )
    assert batch.root_values[final_sample, 1] == 0.0
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


def test_replay_builds_fixed_n_step_values_from_stored_root_values() -> None:
    replay = FIFOReplayBuffer(max_transitions=10, seed=3)
    replay.add(make_trajectory(3, terminated=True))

    batch = replay.sample(
        batch_size=3,
        unroll_steps=3,
        td_steps=2,
        discount=0.5,
    )
    start_zero = int((batch.frames[:, 0, 0, 0, 0] == 0).nonzero().item())

    torch.testing.assert_close(
        batch.value_targets[start_zero],
        torch.tensor([2.5, 3.5, 3.0, 0.0]),
    )
    torch.testing.assert_close(
        batch.value_mask[start_zero],
        torch.tensor([True, True, True, True]),
    )


def test_replay_reconstructs_overlapping_stacks_from_compact_frames() -> None:
    replay = FIFOReplayBuffer(max_transitions=10, seed=1)
    replay.add(make_trajectory(2, stack_size=4))

    batch = replay.sample(batch_size=2, unroll_steps=1)

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


def test_replay_unroll_continues_across_nonterminal_blocks() -> None:
    replay = FIFOReplayBuffer(max_transitions=10, seed=2)
    replay.add(make_trajectory(2, block_id=0))
    replay.add(make_trajectory(3, block_id=1, initial_value=2))

    batch = replay.sample(batch_size=5, unroll_steps=3)
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
        batch.target_mask[crossing_sample],
        torch.tensor([True, True, True, True]),
    )
    torch.testing.assert_close(
        batch.root_values[crossing_sample],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )

    value_batch = replay.sample(
        batch_size=5,
        unroll_steps=3,
        td_steps=2,
        discount=0.5,
    )
    crossing_value_sample = int(
        (value_batch.frames[:, 0, 0, 0, 0] == 1).nonzero().item()
    )
    torch.testing.assert_close(
        value_batch.value_targets[crossing_value_sample],
        torch.tensor([3.25, 3.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        value_batch.value_mask[crossing_value_sample],
        torch.tensor([True, True, False, False]),
    )


def test_replay_rejects_invalid_capacity_and_oversized_samples() -> None:
    with pytest.raises(ValueError, match="positive"):
        FIFOReplayBuffer(0)

    replay = FIFOReplayBuffer(3)
    with pytest.raises(ValueError, match="exceeds"):
        replay.add(make_trajectory(4))

    replay.add(make_trajectory(2))
    with pytest.raises(ValueError, match="cannot sample"):
        replay.sample(3)
