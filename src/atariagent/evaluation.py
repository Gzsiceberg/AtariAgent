"""Checkpoint loading and deterministic Atari policy evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .agent import AtariAgent
from .search import MCTS, MCTSConfig
from .selfplay import Environment


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
    seed: int = 0,
    print_episode_results: bool = False,
) -> EvaluationStats:
    """Evaluate a greedy, noise-free policy and return raw episode scores."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rewards: list[float] = []
    environment: Environment | None = None
    mcts_rng = getattr(getattr(agent, "mcts", None), "rng", None)
    rng_state = mcts_rng.getstate() if mcts_rng is not None else None
    if mcts_rng is not None:
        mcts_rng.seed(seed)
    try:
        environment = environment_factory()
        with tqdm(
            range(episodes),
            desc="Evaluation",
            unit="episode",
            dynamic_ncols=True,
        ) as progress:
            for episode in progress:
                observation, _ = environment.reset(seed=seed + episode)
                episode_reward = 0.0
                terminated = truncated = False
                while not (terminated or truncated):
                    output = agent.act(
                        (observation,),
                        add_exploration_noise=False,
                        temperature=0.0,
                    )
                    observation, reward, terminated, truncated, _ = environment.step(
                        output.actions[0]
                    )
                    episode_reward += float(reward)
                rewards.append(episode_reward)
                progress.set_postfix(reward=f"{episode_reward:.2f}", refresh=False)
                if print_episode_results:
                    progress.write(
                        f"Episode {episode + 1}/{episodes}: "
                        f"reward={episode_reward:.3f}"
                    )
    finally:
        if environment is not None:
            environment.close()
        if mcts_rng is not None:
            mcts_rng.setstate(rng_state)

    return EvaluationStats.from_rewards(tuple(rewards))


def load_agent_checkpoint(
    checkpoint_path: str | Path,
    *,
    action_space_size: int,
    device: torch.device,
) -> tuple[AtariAgent, Mapping[str, Any]]:
    """Build an agent from a training checkpoint and return its saved config."""
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
        num_simulations = int(self_play_config["num_simulations"])
        discount = float(training_config["discount"]) ** int(
            environment_config["frame_skip"]
        )
        lstm_horizon = int(training_config["lstm_horizon"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint {checkpoint_path} has an invalid config") from error

    image_channels = 1 if grayscale else 3
    agent = AtariAgent(
        frame_stack * image_channels,
        action_space_size,
        mcts=MCTS(
            MCTSConfig(
                num_simulations=num_simulations,
                discount=discount,
                value_prefix_horizon=lstm_horizon,
            )
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
