from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import pytest
import torch

from atariagent import AgentOutput
from atariagent.agent import (
    AtariObservation,
    atari_observation_tensor,
    batch_atari_observations,
)
from atariagent.search import SearchResult
from atariagent.selfplay import (
    EpisodeRewardTracker,
    EpisodicLifeEnvironment,
    FULL_EPISODE_DONE_KEY,
    SelfPlayWorker,
)


class DiscreteActionSpace:
    n = 2


class LifeLossEnvironment(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.ale = self
        self.current_lives = 3
        self.reset_calls = 0
        self.step_calls = 0

    def lives(self) -> int:
        return self.current_lives

    def _observation(self) -> np.ndarray:
        return np.full(
            self.observation_space.shape,
            self.step_calls,
            dtype=np.uint8,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_calls += 1
        self.current_lives = 3
        return self._observation(), {}

    def step(self, action):
        self.step_calls += 1
        terminated = False
        reward = 0.0
        if self.step_calls == 1:
            self.current_lives = 2
            reward = 10.0
        elif self.step_calls == 3:
            self.current_lives = 0
            terminated = True
            reward = 20.0
        return self._observation(), reward, terminated, False, {}


class FakeEnvironment:
    def __init__(self, episode_length: int, reward: float = 2.5) -> None:
        self.action_space = DiscreteActionSpace()
        self.episode_length = episode_length
        self.reward = reward
        self.episode = -1
        self.episode_step = 0
        self.reset_seeds: list[int | None] = []
        self.actions: list[int] = []
        self.closed = False

    def _observation(self) -> np.ndarray:
        value = (self.episode * 100 + self.episode_step) % 256
        return np.full((4, 2, 2, 3), value, dtype=np.uint8)

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        self.episode += 1
        self.episode_step = 0
        return self._observation(), {}

    def step(self, action: int):
        self.actions.append(action)
        self.episode_step += 1
        terminated = self.episode_step == self.episode_length
        return self._observation(), self.reward, terminated, False, {}

    def close(self) -> None:
        self.closed = True


class FakeAgent:
    def __init__(self) -> None:
        self.observation_batches: list[tuple[np.ndarray, ...]] = []
        self.kwargs: list[dict[str, object]] = []
        self.search_result_batches: list[tuple[SearchResult, ...]] = []
        self.calls = 0

    def act(
        self,
        observations: Sequence[AtariObservation],
        **kwargs: object,
    ) -> AgentOutput:
        self.observation_batches.append(tuple(observations))
        self.kwargs.append(kwargs)
        self.calls += 1
        results = tuple(
            SearchResult(
                action=1,
                policy_target=(0.25, 0.75),
                root_value=float(self.calls + index),
            )
            for index in range(len(observations))
        )
        self.search_result_batches.append(results)
        return AgentOutput(
            actions=tuple(1 for _ in results),
            search_results=results,
            predicted_values=tuple(
                float(10 * self.calls + index) for index in range(len(results))
            ),
        )


def test_episodic_life_continues_after_life_loss_and_resets_on_game_over() -> None:
    base = LifeLossEnvironment()
    environment = EpisodicLifeEnvironment(base)

    environment.reset(seed=3)
    _, _, terminated, truncated, info = environment.step(1)

    assert terminated
    assert not truncated
    assert info[FULL_EPISODE_DONE_KEY] is False
    assert base.reset_calls == 1

    environment.reset()
    assert base.reset_calls == 1
    assert base.step_calls == 2

    _, _, terminated, _, info = environment.step(1)
    assert terminated
    assert info[FULL_EPISODE_DONE_KEY] is True
    environment.reset()
    assert base.reset_calls == 2


def test_worker_batches_games_and_persists_them_between_runs() -> None:
    agent = FakeAgent()
    first_environment = FakeEnvironment(episode_length=2)
    second_environment = FakeEnvironment(episode_length=6)
    worker = SelfPlayWorker(
        agent,
        environments=[first_environment, second_environment],
        base_seed=10,
    )

    first_run = worker.run(3)

    assert agent.calls == 3
    assert all(
        kwargs
        == {
            "temperature": 1.0,
            "root_noise_temperature": 1.0,
            "gumbel_sampling": True,
        }
        for kwargs in agent.kwargs
    )
    assert all(len(batch) == 2 for batch in agent.observation_batches)
    assert all(obs.shape == (4, 2, 2, 3) for obs in agent.observation_batches[0])
    assert [len(block) for block in first_run[0]] == [2]
    assert first_run[1] == ()
    assert first_run[0][0].terminated
    assert first_run[0][0].episode_id == 0
    assert first_run[0][0].actions == (1, 1)
    assert first_run[0][0].rewards == (1.0, 1.0)
    assert first_run[0][0].raw_rewards == (2.5, 2.5)
    assert first_run[0][0].stack_size == 4
    assert len(first_run[0][0].frames) == 6
    assert all(frame.shape == (3, 2, 2) for frame in first_run[0][0].frames)
    assert [int(frame[0, 0, 0]) for frame in first_run[0][0].frames] == [
        0,
        0,
        0,
        0,
        1,
        2,
    ]
    assert first_run[0][0].search_results[0] is agent.search_result_batches[0][0]
    assert first_run[0][0].target_policy == ((0.25, 0.75), (0.25, 0.75))
    assert first_run[0][0].root_values == (1.0, 2.0)
    assert first_run[0][0].predicted_values == (10.0, 20.0)
    assert first_environment.reset_seeds == [10, None]
    assert second_environment.reset_seeds == [11]

    second_run = worker.run(
        2,
        temperature=0.5,
        root_noise_temperature=0.25,
    )

    assert agent.calls == 5
    assert all(
        kwargs
        == {
            "temperature": 0.5,
            "root_noise_temperature": 0.25,
            "gumbel_sampling": True,
        }
        for kwargs in agent.kwargs[-2:]
    )
    assert [len(block) for block in second_run[0]] == [2]
    assert second_run[0][0].terminated
    assert second_run[0][0].episode_id == 1
    assert second_run[1] == ()

    assert worker.total_vector_steps == 5
    assert worker.total_transitions == 10
    completed = worker.run(1)
    assert [len(block) for block in completed[0]] == [2]
    assert completed[0][0].episode_id == 2
    assert [len(block) for block in completed[1]] == [6]
    assert completed[0][0].terminated
    assert completed[1][0].terminated
    assert [int(frame[0, 0, 0]) for frame in completed[1][0].frames] == [
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert worker.total_vector_steps == 6
    assert worker.total_transitions == 12


def test_worker_waits_for_lookahead_and_keeps_it_in_next_block() -> None:
    worker = SelfPlayWorker(
        FakeAgent(),
        environments=[FakeEnvironment(episode_length=100)],
        trajectory_length=5,
        lookahead_steps=2,
        clip_rewards=False,
    )

    assert worker.run(3)[0] == ()
    assert worker.run(2)[0] == ()
    assert worker.run(1)[0] == ()
    trajectories = worker.run(1)[0]

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert len(trajectory) == 5
    assert trajectory.stored_transition_count == 7
    assert trajectory.lookahead_steps == 2
    assert trajectory.block_id == 0
    assert not trajectory.terminated
    assert trajectory.rewards == (2.5,) * 7

    # Continue until the next block has its own complete lookahead.
    assert worker.run(4)[0] == ()
    future = worker.run(1)[0][0]
    assert len(future) == 5
    assert future.stored_transition_count == 7
    assert future.lookahead_steps == 2
    assert future.block_id == 1
    assert not future.terminated
    assert future.actions[:2] == trajectory.actions[-2:]
    assert future.search_results[:2] == trajectory.search_results[-2:]


def test_terminal_during_lookahead_finalizes_both_blocks() -> None:
    worker = SelfPlayWorker(
        FakeAgent(),
        environments=[FakeEnvironment(episode_length=6, reward=2.5)],
        trajectory_length=5,
        lookahead_steps=2,
        clip_rewards=False,
    )

    trajectories = worker.run(6)[0]

    assert [len(trajectory) for trajectory in trajectories] == [5, 1]
    assert [trajectory.stored_transition_count for trajectory in trajectories] == [
        6,
        1,
    ]
    assert [trajectory.lookahead_steps for trajectory in trajectories] == [1, 0]
    assert all(trajectory.terminated for trajectory in trajectories)
    assert not trajectories[0].full_episode_done
    assert trajectories[1].full_episode_done
    assert EpisodeRewardTracker().add(trajectories) == (15.0,)


def test_episode_reward_tracker_accumulates_raw_rewards_across_blocks() -> None:
    worker = SelfPlayWorker(
        FakeAgent(),
        environments=[FakeEnvironment(episode_length=5, reward=2.5)],
        trajectory_length=2,
        clip_rewards=True,
    )
    tracker = EpisodeRewardTracker()

    first_blocks = worker.run(2)[0]
    final_blocks = worker.run(3)[0]

    assert tracker.add(first_blocks) == ()
    assert tracker.add(final_blocks) == (12.5,)
    worker.close()


def test_episode_reward_tracker_accumulates_across_life_losses() -> None:
    base = LifeLossEnvironment()
    episodic_life = EpisodicLifeEnvironment(base)
    environment = gym.wrappers.FrameStackObservation(episodic_life, stack_size=4)
    worker = SelfPlayWorker(
        FakeAgent(),
        environments=[environment],
        clip_rewards=True,
    )
    tracker = EpisodeRewardTracker()

    life_blocks = worker.run(2)[0]

    assert len(life_blocks) == 2
    assert life_blocks[0].terminated
    assert not life_blocks[0].full_episode_done
    assert life_blocks[1].terminated
    assert life_blocks[1].full_episode_done
    assert [block.episode_id for block in life_blocks] == [0, 0]
    assert tracker.add(life_blocks) == (30.0,)
    worker.close()


def test_worker_closes_all_environments_and_rejects_further_runs() -> None:
    environments = [FakeEnvironment(10), FakeEnvironment(10)]
    worker = SelfPlayWorker(FakeAgent(), environments=environments)

    worker.close()
    worker.close()

    assert all(environment.closed for environment in environments)
    with pytest.raises(RuntimeError, match="closed"):
        worker.run(1)


def test_worker_validates_steps() -> None:
    worker = SelfPlayWorker(FakeAgent(), environments=[FakeEnvironment(10)])

    with pytest.raises(ValueError, match="positive"):
        worker.run(0)
    with pytest.raises(TypeError, match="integer"):
        worker.run(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-negative"):
        SelfPlayWorker(
            FakeAgent(),
            environments=[FakeEnvironment(10)],
            lookahead_steps=-1,
        )


def test_atari_observation_conversion_flattens_frame_and_rgb_channels() -> None:
    observation = np.full((4, 96, 96, 3), 255, dtype=np.uint8)

    converted = atari_observation_tensor(observation)
    batch = batch_atari_observations([observation, observation])

    assert converted.shape == (12, 96, 96)
    assert batch.shape == (2, 12, 96, 96)
    torch.testing.assert_close(converted, torch.ones_like(converted))
