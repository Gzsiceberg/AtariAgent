"""Persistent batched Atari self-play trajectory generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import random
from typing import Protocol

import gymnasium as gym
import numpy as np

from .agent import AgentOutput, AtariObservation
from .search import SearchResult


class DiscreteActionSpace(Protocol):
    """Discrete action-space attributes used by self-play."""

    n: int


class Environment(Protocol):
    """Atari environment operations used by the self-play worker."""

    action_space: DiscreteActionSpace

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[AtariObservation, dict[str, object]]: ...

    def step(
        self, action: int
    ) -> tuple[AtariObservation, float, bool, bool, dict[str, object]]: ...

    def close(self) -> None: ...


class SelfPlayAgent(Protocol):
    """Batched action-selection interface consumed by the worker."""

    def act(
        self,
        observations: Sequence[AtariObservation],
        *,
        add_exploration_noise: bool = False,
        temperature: float = 0.0,
    ) -> AgentOutput: ...


EnvironmentFactory = Callable[[int], Environment]

# Episodic-life training emits an artificial terminal when a life is lost.
# Preserve the underlying environment boundary in ``info`` so score tracking
# can use the same full-episode return definition as evaluation.
FULL_EPISODE_DONE_KEY = "atariagent.full_episode_done"


@dataclass(frozen=True, slots=True)
class GameTrajectory:
    """An immutable compact block from one environment and episode.

    ``frames`` stores the initial frame-stack context once, followed by one
    new processed frame per stored action. A full nonterminal block contains
    ``trajectory_length`` sampleable transitions followed by lookahead
    transitions copied into the next block as well. ``lookahead_steps`` tells
    consumers how many trailing transitions are context rather than replay
    starts. State ``i`` is reconstructed from ``frames[i : i + S]``.
    ``terminated`` may represent an artificial episodic-life terminal, while
    ``full_episode_done`` is true only on the block that owns the full-game
    boundary used by evaluation scoring.
    """

    environment_index: int
    episode_id: int
    block_id: int
    stack_size: int
    frames: tuple[AtariObservation, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    raw_rewards: tuple[float, ...]
    search_results: tuple[SearchResult, ...]
    terminated: bool
    truncated: bool
    full_episode_done: bool
    lookahead_steps: int = 0

    def __post_init__(self) -> None:
        transition_count = len(self.actions)
        if isinstance(self.lookahead_steps, bool) or not isinstance(
            self.lookahead_steps, int
        ):
            raise TypeError("lookahead_steps must be an integer")
        if not 0 <= self.lookahead_steps < transition_count:
            raise ValueError(
                "lookahead_steps must be non-negative and leave at least one "
                "sampleable transition"
            )
        if self.full_episode_done and not (self.terminated or self.truncated):
            raise ValueError(
                "a full episode can end only at a terminal or truncated state"
            )
        if self.stack_size <= 0:
            raise ValueError("stack_size must be positive")
        if len(self.frames) != transition_count + self.stack_size:
            raise ValueError(
                "a trajectory needs stack_size initial frames plus one frame "
                "per action"
            )
        if not self.frames:
            raise ValueError("frames must not be empty")
        frame_shape = self.frames[0].shape
        if any(frame.shape != frame_shape for frame in self.frames):
            raise ValueError("all trajectory frames must have the same shape")
        for values in (
            self.rewards,
            self.raw_rewards,
            self.search_results,
        ):
            if len(values) != transition_count:
                raise ValueError("all transition fields must have equal lengths")
        if transition_count == 0:
            raise ValueError("a finalized trajectory must contain a transition")

    def __len__(self) -> int:
        """Return sampleable transitions, excluding trailing lookahead data."""
        return len(self.actions) - self.lookahead_steps

    @property
    def stored_transition_count(self) -> int:
        """Return all stored transitions, including lookahead context."""
        return len(self.actions)

    def validate_lookahead(self, unroll_steps: int, td_steps: int) -> None:
        """Ensure a complete nonterminal tail covers both replay horizons."""
        required_steps = max(unroll_steps, td_steps)
        if (
            self.lookahead_steps == 0
            or self.terminated
            or self.truncated
            or self.lookahead_steps >= required_steps
        ):
            # Terminal blocks need no bootstrap. A zero-lookahead block is an
            # incomplete flush whose unavailable targets are masked by replay.
            return
        raise ValueError(
            "trajectory lookahead_steps must be greater than or equal to "
            f"unroll_steps and td_steps; got {self.lookahead_steps}, "
            f"unroll_steps={unroll_steps}, td_steps={td_steps}"
        )

    @property
    def target_policy(self) -> tuple[tuple[float, ...], ...]:
        """Normalized MCTS visit distributions used as policy targets."""
        distributions = []
        for result in self.search_results:
            total_visits = sum(result.visit_counts)
            if total_visits <= 0:
                raise ValueError("MCTS search result must contain visited actions")
            distributions.append(
                tuple(count / total_visits for count in result.visit_counts)
            )
        return tuple(distributions)

    @property
    def root_values(self) -> tuple[float, ...]:
        """MCTS root-value targets for every transition."""
        return tuple(result.root_value for result in self.search_results)


class _TrajectoryBuilder:
    def __init__(
        self,
        environment_index: int,
        episode_id: int,
        block_id: int,
        stack_size: int,
        initial_observation: AtariObservation,
    ) -> None:
        self.environment_index = environment_index
        self.episode_id = episode_id
        self.block_id = block_id
        self.stack_size = stack_size
        self.frames = list(
            _stacked_observation_frames(initial_observation, stack_size)
        )
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.raw_rewards: list[float] = []
        self.search_results: list[SearchResult] = []
        self._sampleable_transitions: int | None = None

    @property
    def lookahead_steps(self) -> int:
        if self._sampleable_transitions is None:
            return 0
        return len(self.actions) - self._sampleable_transitions

    def freeze(self) -> None:
        """Mark the current transitions as replay starts before adding context."""
        if self._sampleable_transitions is not None:
            raise RuntimeError("trajectory builder is already frozen")
        if not self.actions:
            raise RuntimeError("cannot freeze an empty trajectory builder")
        self._sampleable_transitions = len(self.actions)

    def append(
        self,
        *,
        action: int,
        observation: AtariObservation,
        reward: float,
        raw_reward: float,
        search_result: SearchResult,
    ) -> None:
        self.actions.append(int(action))
        next_frames = _stacked_observation_frames(observation, self.stack_size)
        self.frames.append(next_frames[-1])
        self.rewards.append(float(reward))
        self.raw_rewards.append(float(raw_reward))
        self.search_results.append(search_result)

    def __len__(self) -> int:
        return len(self.actions)

    def finalize(
        self,
        *,
        terminated: bool,
        truncated: bool,
        full_episode_done: bool,
    ) -> GameTrajectory:
        return GameTrajectory(
            environment_index=self.environment_index,
            episode_id=self.episode_id,
            block_id=self.block_id,
            stack_size=self.stack_size,
            frames=tuple(self.frames),
            actions=tuple(self.actions),
            rewards=tuple(self.rewards),
            raw_rewards=tuple(self.raw_rewards),
            search_results=tuple(self.search_results),
            terminated=terminated,
            truncated=truncated,
            full_episode_done=full_episode_done,
            lookahead_steps=self.lookahead_steps,
        )


class EpisodicLifeEnvironment(gym.Wrapper):
    """Expose life losses as terminals without resetting the underlying game."""

    def __init__(self, environment: gym.Env) -> None:
        super().__init__(environment)
        self._lives = 0
        self._was_real_done = True

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._was_real_done = bool(terminated or truncated)
        info = dict(info)
        info[FULL_EPISODE_DONE_KEY] = self._was_real_done
        lives = int(self.env.unwrapped.ale.lives())
        if 0 < lives < self._lives:
            terminated = True
        self._lives = lives
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options=None):
        if self._was_real_done:
            observation, info = self.env.reset(seed=seed, options=options)
        else:
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(seed=seed, options=options)
        self._lives = int(self.env.unwrapped.ale.lives())
        return observation, info


def make_atari_environment(
    env_id: str,
    *,
    frame_stack: int = 4,
    frame_skip: int = 4,
    screen_size: int = 96,
    max_episode_steps: int = 3000,
    terminal_on_life_loss: bool = False,
    grayscale_obs: bool = False,
    render_mode: str | None = None,
) -> Environment:
    """Create an Atari environment matching the project's training setup."""
    if frame_stack <= 0:
        raise ValueError("frame_stack must be positive")
    if frame_skip <= 0:
        raise ValueError("frame_skip must be positive")
    if screen_size <= 0:
        raise ValueError("screen_size must be positive")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")

    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import (
        AtariPreprocessing,
        FrameStackObservation,
        TimeLimit,
    )

    gym.register_envs(ale_py)
    base_environment = gym.make(
        env_id,
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
    )
    environment = AtariPreprocessing(
        base_environment,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=grayscale_obs,
        scale_obs=False,
    )
    environment = TimeLimit(environment, max_episode_steps=max_episode_steps)
    if terminal_on_life_loss:
        environment = EpisodicLifeEnvironment(environment)
    return FrameStackObservation(environment, stack_size=frame_stack)


class EpisodeRewardTracker:
    """Emit full-game raw returns using the same score definition as evaluation."""

    def __init__(self) -> None:
        self._partial_rewards: dict[tuple[int, int], float] = {}

    def add(self, trajectories: Sequence[GameTrajectory]) -> tuple[float, ...]:
        """Accumulate life/block rewards and return completed full-game scores."""
        completed_rewards: list[float] = []
        for trajectory in trajectories:
            episode_key = (trajectory.environment_index, trajectory.episode_id)
            episode_reward = self._partial_rewards.get(episode_key, 0.0) + sum(
                trajectory.raw_rewards[: len(trajectory)]
            )
            if trajectory.full_episode_done:
                completed_rewards.append(episode_reward)
                self._partial_rewards.pop(episode_key, None)
            else:
                self._partial_rewards[episode_key] = episode_reward
        return tuple(completed_rewards)


class SelfPlayWorker:
    """Generate replay-ready trajectory blocks from persistent Atari games.

    ``run(steps)`` advances every environment by exactly ``steps`` transitions,
    batching one agent call across all environments at each step. At
    ``trajectory_length`` the block remains open for ``lookahead_steps`` more
    transitions. Those trailing transitions are also retained as the start of
    the next block. Environment terminals (including configured life losses)
    finalize every open block immediately. Partial builders persist across
    calls so worker scheduling does not create artificial replay boundaries.
    """

    def __init__(
        self,
        agent: SelfPlayAgent,
        *,
        env_id: str = "ALE/Pong-v5",
        num_envs: int | None = None,
        environments: Sequence[Environment] | None = None,
        environment_factory: EnvironmentFactory | None = None,
        frame_stack: int = 4,
        frame_skip: int = 4,
        screen_size: int = 96,
        max_episode_steps: int = 3000,
        trajectory_length: int = 400,
        lookahead_steps: int = 5,
        base_seed: int = 0,
        clip_rewards: bool = True,
        add_exploration_noise: bool = True,
        temperature: float = 1.0,
    ) -> None:
        if environments is not None and environment_factory is not None:
            raise ValueError(
                "pass either environments or environment_factory, not both"
            )
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if isinstance(trajectory_length, bool) or not isinstance(
            trajectory_length, int
        ):
            raise TypeError("trajectory_length must be an integer")
        if trajectory_length <= 0:
            raise ValueError("trajectory_length must be positive")
        if isinstance(lookahead_steps, bool) or not isinstance(
            lookahead_steps, int
        ):
            raise TypeError("lookahead_steps must be an integer")
        if lookahead_steps < 0:
            raise ValueError("lookahead_steps must be non-negative")

        if environments is not None:
            self.environments = list(environments)
            if not self.environments:
                raise ValueError("environments must not be empty")
            if num_envs is not None and num_envs != len(self.environments):
                raise ValueError("num_envs must match the supplied environments")
            num_envs = len(self.environments)
        else:
            num_envs = 4 if num_envs is None else num_envs
            if num_envs <= 0:
                raise ValueError("num_envs must be positive")
            factory = environment_factory
            if factory is None:
                factory = lambda _seed: make_atari_environment(
                    env_id,
                    frame_stack=frame_stack,
                    frame_skip=frame_skip,
                    screen_size=screen_size,
                    max_episode_steps=max_episode_steps,
                )
            self.environments = [
                factory(base_seed + index) for index in range(num_envs)
            ]

        self.agent = agent
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.frame_stack = frame_stack
        self.trajectory_length = trajectory_length
        self.lookahead_steps = lookahead_steps
        self.clip_rewards = clip_rewards
        self.add_exploration_noise = add_exploration_noise
        self.temperature = float(temperature)
        self._rng = random.Random(base_seed)

        self.total_vector_steps = 0
        self.total_transitions = 0
        self._observations: list[AtariObservation] = []
        self._builders: list[list[_TrajectoryBuilder]] = []
        self._episode_ids = [0] * self.num_envs
        self._next_block_ids = [0] * self.num_envs
        self._initialized = False
        self._closed = False

    def run(
        self,
        steps: int,
        *,
        temperature: float | None = None,
        random_actions: bool = False,
    ) -> tuple[tuple[GameTrajectory, ...], ...]:
        """Advance each game and return blocks grouped by game.

        ``random_actions`` uses a seeded uniform behavior policy while retaining
        MCTS root values and storing a uniform policy target. This matches the
        EfficientZero replay warmup behavior.
        """
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not isinstance(random_actions, bool):
            raise TypeError("random_actions must be a boolean")
        active_temperature = (
            self.temperature if temperature is None else temperature
        )
        if not np.isfinite(active_temperature) or active_temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if self._closed:
            raise RuntimeError("cannot run a closed self-play worker")

        self._ensure_initialized()
        completed: list[list[GameTrajectory]] = [
            [] for _ in range(self.num_envs)
        ]
        builders = self._builders

        for _ in range(steps):
            agent_output = self.agent.act(
                self._observations,
                add_exploration_noise=self.add_exploration_noise,
                temperature=float(active_temperature),
            )
            if random_actions:
                agent_output = self._uniform_behavior_output(agent_output)
            self._validate_agent_output(agent_output)

            for index, environment in enumerate(self.environments):
                action = agent_output.actions[index]
                next_observation, raw_reward, terminated, truncated, info = (
                    environment.step(action)
                )
                raw_reward = float(raw_reward)
                reward = (
                    float(np.sign(raw_reward))
                    if self.clip_rewards
                    else raw_reward
                )
                # Frozen builders receive the transition as lookahead context,
                # while the active builder owns it as a future replay start.
                for builder in builders[index]:
                    builder.append(
                        action=action,
                        observation=next_observation,
                        reward=reward,
                        raw_reward=raw_reward,
                        search_result=agent_output.search_results[index],
                    )
                self._observations[index] = _copy_observation(next_observation)
                self.total_transitions += 1

                episode_done = bool(terminated or truncated)
                full_episode_done = bool(
                    info.get(FULL_EPISODE_DONE_KEY, episode_done)
                )
                if full_episode_done and not episode_done:
                    raise ValueError(
                        f"{FULL_EPISODE_DONE_KEY} requires a terminal or "
                        "truncated transition"
                    )
                if episode_done:
                    # A terminal in the bootstrap tail also terminates an older
                    # stored block, but only the active block owns that episode
                    # boundary for score accounting.
                    final_builder = builders[index][-1]
                    completed[index].extend(
                        builder.finalize(
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            full_episode_done=(
                                full_episode_done and builder is final_builder
                            ),
                        )
                        for builder in builders[index]
                        if len(builder) > 0
                    )
                    self._next_block_ids[index] += 1
                    if full_episode_done:
                        self._episode_ids[index] += 1
                    self._observations[index] = self._reset_environment(
                        index, seed=None
                    )
                    builders[index] = [self._new_builder(index)]
                    continue

                active_builder = builders[index][-1]
                if len(active_builder) >= self.trajectory_length:
                    active_builder.freeze()
                    self._next_block_ids[index] += 1
                    builders[index].append(self._new_builder(index))

                ready = [
                    builder
                    for builder in builders[index][:-1]
                    if builder.lookahead_steps >= self.lookahead_steps
                ]
                completed[index].extend(
                    builder.finalize(
                        terminated=False,
                        truncated=False,
                        full_episode_done=False,
                    )
                    for builder in ready
                )
                if ready:
                    ready_ids = {id(builder) for builder in ready}
                    builders[index] = [
                        builder
                        for builder in builders[index]
                        if id(builder) not in ready_ids
                    ]

            self.total_vector_steps += 1

        return tuple(tuple(blocks) for blocks in completed)

    def flush(self) -> tuple[tuple[GameTrajectory, ...], ...]:
        """Finalize non-empty partial blocks without resetting environments.

        This is intended for inspection or shutdown. Training collection should
        normally keep partial builders across calls to avoid artificial block
        boundaries.
        """
        if self._closed:
            raise RuntimeError("cannot flush a closed self-play worker")
        self._ensure_initialized()
        completed: list[list[GameTrajectory]] = [
            [] for _ in range(self.num_envs)
        ]
        for index, builders in enumerate(self._builders):
            active_builder = builders[-1]
            completed[index].extend(
                builder.finalize(
                    terminated=False,
                    truncated=False,
                    full_episode_done=False,
                )
                for builder in builders
                if len(builder) > 0
            )
            # A non-empty active block consumed its reserved identity. An empty
            # one was created only to retain a just-completed block's tail and
            # can safely reuse the same identity after flushing.
            if len(active_builder) > 0:
                self._next_block_ids[index] += 1
            self._builders[index] = [self._new_builder(index)]
        return tuple(tuple(blocks) for blocks in completed)

    def close(self) -> None:
        """Close every environment. Calling this method repeatedly is safe."""
        if self._closed:
            return
        for environment in self.environments:
            environment.close()
        self._closed = True

    def __enter__(self) -> SelfPlayWorker:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._observations = [
            self._reset_environment(
                index,
                seed=self.base_seed + index,
            )
            for index in range(self.num_envs)
        ]
        self._builders = [
            [self._new_builder(index)] for index in range(self.num_envs)
        ]
        self._initialized = True

    def _reset_environment(
        self, index: int, *, seed: int | None
    ) -> AtariObservation:
        observation, _ = self.environments[index].reset(seed=seed)
        return _copy_observation(observation)

    def _new_builder(self, index: int) -> _TrajectoryBuilder:
        return _TrajectoryBuilder(
            environment_index=index,
            episode_id=self._episode_ids[index],
            block_id=self._next_block_ids[index],
            stack_size=self.frame_stack,
            initial_observation=self._observations[index],
        )

    def _uniform_behavior_output(self, output: AgentOutput) -> AgentOutput:
        """Replace actions and visit targets with a seeded uniform policy."""
        actions: list[int] = []
        results: list[SearchResult] = []
        for result, environment in zip(
            output.search_results, self.environments, strict=True
        ):
            action_count = int(environment.action_space.n)
            action = self._rng.randrange(action_count)
            actions.append(action)
            results.append(
                SearchResult(
                    action=action,
                    visit_counts=tuple(1 for _ in range(action_count)),
                    root_value=result.root_value,
                )
            )
        return AgentOutput(actions=tuple(actions), search_results=tuple(results))

    def _validate_agent_output(self, output: AgentOutput) -> None:
        if len(output.actions) != self.num_envs:
            raise ValueError("agent must return one action per environment")
        if len(output.search_results) != self.num_envs:
            raise ValueError("agent must return one search result per environment")
        for index, (action, result, environment) in enumerate(
            zip(
                output.actions,
                output.search_results,
                self.environments,
                strict=True,
            )
        ):
            action_count = getattr(environment.action_space, "n", None)
            if action_count is not None and action not in range(action_count):
                raise ValueError(
                    f"agent returned invalid action {action} for environment {index}"
                )
            if result.action != action:
                raise ValueError("agent action must match its MCTS search result")


def _copy_observation(observation: AtariObservation) -> AtariObservation:
    return observation.copy()


def _stacked_observation_frames(
    observation: AtariObservation, stack_size: int
) -> tuple[AtariObservation, ...]:
    """Split a Gym frame stack into copied channel-first uint8 frames."""
    stacked = np.asarray(observation, dtype=np.uint8)
    if stacked.ndim == 4:
        if stacked.shape[0] != stack_size:
            raise ValueError("observation stack axis does not match stack_size")
        return tuple(
            np.moveaxis(frame, -1, 0).copy() for frame in stacked
        )
    if stacked.ndim == 3 and stacked.shape[0] == stack_size:
        return tuple(frame[np.newaxis, ...].copy() for frame in stacked)
    if stacked.ndim == 3 and stacked.shape[0] % stack_size == 0:
        channels = stacked.shape[0] // stack_size
        return tuple(
            stacked[index * channels : (index + 1) * channels].copy()
            for index in range(stack_size)
        )
    raise ValueError(
        "observation must be a frame stack in stack-first or flattened "
        "channel-first format"
    )


__all__ = [
    "DiscreteActionSpace",
    "Environment",
    "EnvironmentFactory",
    "EpisodicLifeEnvironment",
    "FULL_EPISODE_DONE_KEY",
    "GameTrajectory",
    "SelfPlayAgent",
    "SelfPlayWorker",
    "make_atari_environment",
]
