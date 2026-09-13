"""Checkpoint loading and deterministic Atari policy evaluation utilities."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .agent import AtariAgent
from .search import SearchConfig, efficientzero_atari_gumbel_settings
from .selfplay import Environment
from .typecheck import runtime_typechecking_enabled, set_runtime_typechecking


@dataclass(frozen=True, slots=True)
class EvaluationStats:
    """Episode rewards and their summary statistics."""

    rewards: tuple[float, ...]
    mean: float
    median: float
    std: float

    @classmethod
    def from_rewards(cls, rewards: tuple[float, ...]) -> EvaluationStats:
        if not rewards:
            raise ValueError("rewards must not be empty")
        values = np.asarray(rewards, dtype=np.float64)
        return cls(
            rewards=tuple(float(reward) for reward in values),
            mean=float(values.mean()),
            median=float(np.median(values)),
            std=float(values.std()),
        )


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One checkpoint's evaluation result."""

    update: int
    checkpoint: str
    rewards: tuple[float, ...]
    mean: float
    median: float
    std: float

    @classmethod
    def create(
        cls, update: int, checkpoint: Path, stats: EvaluationStats
    ) -> EvaluationRecord:
        return cls(
            update=update,
            checkpoint=str(checkpoint),
            rewards=stats.rewards,
            mean=stats.mean,
            median=stats.median,
            std=stats.std,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "update": self.update,
            "checkpoint": self.checkpoint,
            "rewards": list(self.rewards),
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
        }


def evaluate_agent(
    agent: AtariAgent,
    environment_factory: Callable[[], Environment],
    *,
    episodes: int,
    num_envs: int = 16,
    seed: int = 0,
    print_episode_results: bool = False,
) -> EvaluationStats:
    """Evaluate a greedy policy, batching inference across parallel games.

    Environments that finish early are immediately assigned the next episode,
    so up to ``num_envs`` games remain active until all episodes are complete.
    Returned rewards retain episode/seed order rather than completion order.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an integer")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int):
        raise TypeError("num_envs must be an integer")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    rewards = [0.0] * episodes
    environments: list[Environment] = []
    created_environments: list[Environment] = []
    search = getattr(agent, "search", getattr(agent, "mcts", None))
    search_rng = getattr(search, "rng", None)
    rng_state = search_rng.getstate() if search_rng is not None else None
    typechecking_was_enabled = runtime_typechecking_enabled()
    if search_rng is not None:
        search_rng.seed(seed)
    # Evaluation repeatedly executes the same validated model interfaces in a
    # hot tree-search loop. Keep annotations, but avoid type-check overhead.
    set_runtime_typechecking(False)
    try:
        active_count = min(num_envs, episodes)
        for _ in range(active_count):
            created_environments.append(environment_factory())
        environments = created_environments.copy()
        observations = []
        episode_indices = list(range(active_count))
        episode_rewards = [0.0] * active_count
        for environment, episode in zip(environments, episode_indices, strict=True):
            observation, _ = environment.reset(seed=seed + episode)
            observations.append(observation)

        next_episode = active_count
        with tqdm(
            total=episodes,
            desc="Evaluation",
            unit="episode",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ) as progress:
            while environments:
                output = agent.act(
                    observations,
                    temperature=0.0,
                    root_noise_temperature=0.0,
                    gumbel_sampling=False,
                )
                if len(output.actions) != len(environments):
                    raise ValueError(
                        "agent returned a different number of actions than "
                        "active evaluation environments"
                    )

                next_environments: list[Environment] = []
                next_observations = []
                next_episode_indices: list[int] = []
                next_episode_rewards: list[float] = []
                for index, environment in enumerate(environments):
                    observation, reward, terminated, truncated, _ = environment.step(
                        output.actions[index]
                    )
                    episode_reward = episode_rewards[index] + float(reward)
                    episode = episode_indices[index]
                    if terminated or truncated:
                        rewards[episode] = episode_reward
                        progress.update()
                        progress.set_postfix(
                            reward=f"{episode_reward:.2f}", refresh=False
                        )
                        if print_episode_results:
                            progress.write(
                                f"Episode {episode + 1}/{episodes}: "
                                f"reward={episode_reward:.3f}"
                            )
                        if next_episode >= episodes:
                            continue
                        episode = next_episode
                        next_episode += 1
                        observation, _ = environment.reset(seed=seed + episode)
                        episode_reward = 0.0

                    next_environments.append(environment)
                    next_observations.append(observation)
                    next_episode_indices.append(episode)
                    next_episode_rewards.append(episode_reward)

                environments = next_environments
                observations = next_observations
                episode_indices = next_episode_indices
                episode_rewards = next_episode_rewards
    finally:
        for environment in created_environments:
            environment.close()
        if search_rng is not None:
            search_rng.setstate(rng_state)
        set_runtime_typechecking(typechecking_was_enabled)

    return EvaluationStats.from_rewards(tuple(rewards))


def load_agent_checkpoint(
    checkpoint_path: str | Path,
    *,
    action_space_size: int,
    device: torch.device,
    search_algorithm_override: str | None = None,
    num_simulations_override: int | None = None,
) -> tuple[AtariAgent, Mapping[str, Any]]:
    """Build an agent from a checkpoint, optionally overriding its search."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"checkpoint {checkpoint_path} has no training config")

    try:
        environment_config = config["environment"]
        self_play_config = config["self_play"]
        training_config = config["training"]
        frame_stack = int(environment_config["frame_stack"])
        grayscale = bool(environment_config["grayscale"])
        num_simulations = (
            int(num_simulations_override)
            if num_simulations_override is not None
            else int(self_play_config["num_simulations"])
        )
        search_algorithm = (
            str(search_algorithm_override)
            if search_algorithm_override is not None
            else str(self_play_config.get("search_algorithm", "puct"))
        )
        num_top_actions = 4  # Unused by PUCT.
        c_visit = float(self_play_config.get("c_visit", 50.0))
        c_scale = float(self_play_config.get("c_scale", 0.1))
        discount = float(training_config["discount"]) ** int(
            environment_config["frame_skip"]
        )
        lstm_horizon = int(training_config["lstm_horizon"])
        precision = str(training_config.get("precision", "fp32"))
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if search_algorithm == "gumbel":
            num_simulations, num_top_actions = efficientzero_atari_gumbel_settings(
                action_space_size,
                num_simulations,
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"checkpoint {checkpoint_path} has an invalid config"
        ) from error

    print(
        "Evaluation search: "
        f"num_simulations={num_simulations}, "
        f"num_top_actions={num_top_actions}, precision={precision}"
    )

    image_channels = 1 if grayscale else 3
    agent = AtariAgent(
        frame_stack * image_channels,
        action_space_size,
        precision=precision,
        search_config=SearchConfig(
            num_simulations=num_simulations,
            discount=discount,
            value_prefix_horizon=lstm_horizon,
            search_algorithm=search_algorithm,
            num_top_actions=num_top_actions,
            c_visit=c_visit,
            c_scale=c_scale,
        ),
    ).to(device)
    for key, network in (
        ("representation", agent.representation_network),
        ("dynamics", agent.dynamics_network),
        ("prediction", agent.prediction_network),
    ):
        if key not in checkpoint:
            raise ValueError(f"checkpoint {checkpoint_path} is missing {key}")
        network.load_state_dict(checkpoint[key])
    agent.eval()
    return agent, config


def write_evaluation_history(
    path: str | Path,
    records: list[EvaluationRecord],
    *,
    environment_id: str,
) -> None:
    """Atomically write machine-readable evaluation history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "environment": environment_id,
                "evaluations": [record.as_dict() for record in records],
            },
            indent=2,
        )
        + "\n"
    )
    temporary_path.replace(path)


def plot_evaluation_history(
    path: str | Path,
    records: list[EvaluationRecord],
    *,
    title: str,
) -> None:
    """Plot mean and median score with a one-standard-deviation band."""
    if not records:
        raise ValueError("records must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    updates = np.asarray([record.update for record in records])
    means = np.asarray([record.mean for record in records])
    medians = np.asarray([record.median for record in records])
    stds = np.asarray([record.std for record in records])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(updates, means, color="tab:blue", label="Mean reward")
    axis.plot(
        updates,
        medians,
        color="tab:orange",
        linestyle="--",
        alpha=0.8,
        label="Median reward",
    )
    axis.fill_between(
        updates,
        means - stds,
        means + stds,
        color="tab:blue",
        alpha=0.2,
        label="Mean ± std",
    )
    axis.set_title(title)
    axis.set_xlabel("Training updates")
    axis.set_ylabel("Episode reward")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


__all__ = [
    "EvaluationRecord",
    "EvaluationStats",
    "evaluate_agent",
    "load_agent_checkpoint",
    "plot_evaluation_history",
    "write_evaluation_history",
]
