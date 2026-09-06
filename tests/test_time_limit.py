"""Time-limit semantics through the production Atari environment factory."""

import gymnasium as gym
import numpy as np
import pytest

from atariagent.selfplay import FULL_EPISODE_DONE_KEY, make_atari_environment


class LifeBudgetEnvironment(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.ale = self
        self.reset_calls = 0
        self.steps = 0
        self.current_lives = 3

    def lives(self):
        return self.current_lives

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_calls += 1
        self.steps = 0
        self.current_lives = 3
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        if self.steps == 1:
            self.current_lives = 2
        return (
            np.full(self.observation_space.shape, self.steps, dtype=np.uint8),
            1.0,
            False,
            False,
            {},
        )


@pytest.fixture
def base_environment(monkeypatch):
    base = LifeBudgetEnvironment()
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: base)
    # Exercise real wrapper composition without Atari ROMs or preprocessing.
    monkeypatch.setattr(
        gym.wrappers, "AtariPreprocessing", lambda env, **kwargs: env
    )
    return base


def test_life_reset_time_budget_and_timeout_reset(base_environment):
    remaining_steps = 3
    env = make_atari_environment(
        "mock", max_episode_steps=3,
        terminal_on_life_loss=True,
    )
    try:
        env.reset(seed=7)
        _, _, terminated, truncated, info = env.step(1)
        assert terminated and not truncated
        assert not info[FULL_EPISODE_DONE_KEY]

        env.reset()  # Continue the same game with one no-op.
        assert base_environment.reset_calls == 1
        assert base_environment.steps == 2

        for step in range(remaining_steps):
            _, _, terminated, truncated, info = env.step(1)
            assert not terminated
            assert truncated == (step == remaining_steps - 1)
            assert info[FULL_EPISODE_DONE_KEY] == truncated

        env.reset()  # A timeout must reset the game, not just advance a no-op.
        assert base_environment.reset_calls == 2
        assert base_environment.steps == 0
    finally:
        env.close()


def test_without_episodic_life_uses_full_game_budget(base_environment):
    env = make_atari_environment(
        "mock", max_episode_steps=3,
        terminal_on_life_loss=False,
    )
    try:
        env.reset()
        for step in range(3):
            _, _, terminated, truncated, _ = env.step(1)
            assert not terminated  # Life loss is not a training terminal.
            assert truncated == (step == 2)
        env.reset()
        assert base_environment.reset_calls == 2
    finally:
        env.close()
