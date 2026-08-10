#!/usr/bin/env python3
"""Train AtariAgent with MuZero losses on FIFO self-play replay."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from atariagent import AtariAgent, FIFOReplayBuffer, GameTrajectory, SelfPlayWorker
from atariagent.search import MCTSConfig
from atariagent.selfplay import Environment, make_atari_environment
from atariagent.training import MuZeroTrainer
from atariagent.training.muzero_config import (
    TrainMuZeroConfig,
    register_train_muzero_config,
)


register_train_muzero_config()


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def flatten_trajectories(
    trajectories_by_environment: tuple[tuple[GameTrajectory, ...], ...],
) -> Iterable[GameTrajectory]:
    """Flatten grouped worker output without changing block order."""
    for trajectories in trajectories_by_environment:
        yield from trajectories


def create_environments(config: TrainMuZeroConfig) -> list[Environment]:
    """Create identically preprocessed Atari environments."""
    environments: list[Environment] = []
    try:
        for _ in range(config.self_play.num_envs):
            environments.append(
                make_atari_environment(
                    config.environment.id,
                    frame_stack=config.environment.frame_stack,
                    frame_skip=config.environment.frame_skip,
                    screen_size=config.environment.screen_size,
                    max_episode_steps=config.environment.max_episode_steps,
                    grayscale_obs=config.environment.grayscale,
                )
            )
    except Exception:
        for environment in environments:
            environment.close()
        raise
    return environments


def save_checkpoint(
    path: Path,
    *,
    agent: AtariAgent,
    trainer: MuZeroTrainer,
    update: int,
    config: TrainMuZeroConfig,
) -> None:
    """Persist every trainable component and optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "update": update,
            "representation": agent.representation_network.state_dict(),
            "dynamics": agent.dynamics_network.state_dict(),
            "prediction": agent.prediction_network.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "config": OmegaConf.to_container(config, resolve=True),
        },
        path,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="train_muzero")
def main(config: TrainMuZeroConfig) -> None:
    """Alternate self-play collection with updates sampled from replay."""
    if config.training.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for batch normalization")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.training.device)
    environments = create_environments(config)

    try:
        action_space_size = int(environments[0].action_space.n)
        if any(
            int(environment.action_space.n) != action_space_size
            for environment in environments
        ):
            raise ValueError("all environments must have the same action space")

        image_channels = 1 if config.environment.grayscale else 3
        discount = config.training.discount ** config.environment.frame_skip
        agent = AtariAgent(
            config.environment.frame_stack * image_channels,
            action_space_size,
            mcts_config=MCTSConfig(
                num_simulations=config.self_play.num_simulations,
                discount=discount,
                value_prefix_horizon=config.training.lstm_horizon,
            ),
        ).to(device)
        trainer = MuZeroTrainer(
            agent.representation_network,
            agent.dynamics_network,
            agent.prediction_network,
            learning_rate=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
            lr_warmup_steps=config.training.lr_warmup_steps,
            lr_decay_rate=config.training.lr_decay_rate,
            lr_decay_steps=config.training.lr_decay_steps,
            unroll_steps=config.training.unroll_steps,
            lstm_horizon=config.training.lstm_horizon,
            policy_weight=config.loss.policy_weight,
            value_weight=config.loss.value_weight,
            reward_weight=config.loss.reward_weight,
            max_gradient_norm=config.training.max_gradient_norm,
        )
        replay = FIFOReplayBuffer(
            config.replay.max_transitions, seed=config.seed
        )

        update = 0
        collection_iteration = 0
        with SelfPlayWorker(
            agent,
            environments=environments,
            frame_stack=config.environment.frame_stack,
            trajectory_length=config.self_play.trajectory_length,
            base_seed=config.seed,
            clip_rewards=config.self_play.clip_rewards,
            add_exploration_noise=config.self_play.add_exploration_noise,
            temperature=config.self_play.temperature,
        ) as worker:
            while update < config.training.steps:
                collection_iteration += 1
                grouped = worker.run(config.self_play.steps_per_iteration)
                trajectories = tuple(flatten_trajectories(grouped))
                insertion = replay.extend(trajectories)
                print(
                    f"collection={collection_iteration:04d} "
                    f"added={insertion.added_transitions} "
                    f"replay={len(replay)}/{replay.max_transitions}"
                )

                if len(replay) < max(
                    config.replay.warmup_transitions,
                    config.training.batch_size,
                ):
                    continue

                updates = min(
                    config.training.updates_per_iteration,
                    config.training.steps - update,
                )
                for _ in range(updates):
                    batch = replay.sample(
                        config.training.batch_size,
                        unroll_steps=config.training.unroll_steps,
                        td_steps=config.training.td_steps,
                        discount=discount,
                    ).to(device)
                    metrics = trainer.train_step(batch)
                    update += 1
                    if update == 1 or update % config.training.log_every == 0:
                        print(
                            f"update={update:06d} loss={metrics.loss:.4f} "
                            f"policy={metrics.policy_loss:.4f} "
                            f"value={metrics.value_loss:.4f} "
                            f"reward={metrics.reward_loss:.4f} "
                            f"grad_norm={metrics.gradient_norm:.4f} "
                            f"lr={metrics.learning_rate:.6f}"
                        )
                    if (
                        config.checkpoint.every > 0
                        and update % config.checkpoint.every == 0
                    ):
                        save_checkpoint(
                            Path(config.checkpoint.path),
                            agent=agent,
                            trainer=trainer,
                            update=update,
                            config=config,
                        )

        save_checkpoint(
            Path(config.checkpoint.path),
            agent=agent,
            trainer=trainer,
            update=update,
            config=config,
        )
    except Exception:
        # The worker takes ownership only after its context has been entered.
        for environment in environments:
            environment.close()
        raise


if __name__ == "__main__":
    main()
