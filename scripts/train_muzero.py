#!/usr/bin/env python3
"""Train AtariAgent with MuZero losses on FIFO self-play replay."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import shutil
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
from rich import print as rich_print
import torch
from tqdm.auto import tqdm

from atariagent import AtariAgent, FIFOReplayBuffer, GameTrajectory, SelfPlayWorker
from atariagent.evaluation import (
    EvaluationRecord,
    evaluate_agent,
    plot_evaluation_history,
    write_evaluation_history,
)
from atariagent.search import MCTS, MCTSConfig
from atariagent.selfplay import Environment, make_atari_environment
from atariagent.training import (
    MuZeroTrainer,
    representative_checkpoint_path,
    representative_checkpoint_updates,
)
from atariagent.training.muzero_config import (
    TrainMuZeroConfig,
    linear_priority_beta,
    next_collection_vector_steps,
    register_train_muzero_config,
    visit_softmax_temperature,
)


register_train_muzero_config()


def log(message: str) -> None:
    """Print Rich markup without corrupting active tqdm progress bars."""
    with tqdm.external_write_mode():
        rich_print(message)


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
                    terminal_on_life_loss=config.environment.episodic_life,
                )
            )
    except Exception:
        for environment in environments:
            environment.close()
        raise
    return environments


def create_evaluation_environment(config: TrainMuZeroConfig) -> Environment:
    """Create a full-episode Atari environment for policy evaluation."""
    return make_atari_environment(
        config.environment.id,
        frame_stack=config.environment.frame_stack,
        frame_skip=config.environment.frame_skip,
        screen_size=config.environment.screen_size,
        max_episode_steps=config.environment.max_episode_steps,
        grayscale_obs=config.environment.grayscale,
        terminal_on_life_loss=False,
    )


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
    if config.training.steps <= 0:
        raise ValueError("training.steps must be positive")
    if config.training.final_steps < 0:
        raise ValueError("training.final_steps must be non-negative")
    if config.training.updates_per_iteration <= 0:
        raise ValueError("updates_per_iteration must be positive")
    if config.training.log_every <= 0:
        raise ValueError("log_every must be positive")
    if config.checkpoint.every < 0:
        raise ValueError("checkpoint.every must be non-negative")
    if config.checkpoint.keep_representative <= 0:
        raise ValueError("checkpoint.keep_representative must be positive")
    if config.evaluation.enabled and config.evaluation.episodes <= 0:
        raise ValueError("evaluation.episodes must be positive")
    if config.self_play.num_envs <= 0:
        raise ValueError("self_play.num_envs must be positive")
    if config.self_play.total_transitions <= 0:
        raise ValueError("self_play.total_transitions must be positive")
    if config.self_play.steps_per_iteration <= 0:
        raise ValueError("self_play.steps_per_iteration must be positive")
    if config.self_play.total_transitions % config.self_play.num_envs != 0:
        raise ValueError(
            "self_play.total_transitions must be divisible by self_play.num_envs"
        )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
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
        mcts = MCTS(
            MCTSConfig(
                num_simulations=config.self_play.num_simulations,
                discount=discount,
                value_prefix_horizon=config.training.lstm_horizon,
            ),
            rng=random.Random(config.seed),
        )
        agent = AtariAgent(
            config.environment.frame_stack * image_channels,
            action_space_size,
            mcts=mcts,
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
            priority_epsilon=config.replay.priority_epsilon,
        )
        replay = FIFOReplayBuffer(
            config.replay.max_transitions,
            seed=config.seed,
            priority_alpha=config.replay.priority_alpha,
        )

        total_updates = config.training.steps + config.training.final_steps
        representative_updates = set(
            representative_checkpoint_updates(
                total_updates, config.checkpoint.keep_representative
            )
        )
        evaluation_records: list[EvaluationRecord] = []
        checkpointed_updates: set[int] = set()
        latest_checkpoint_path = Path(config.checkpoint.path)
        update = 0
        if config.evaluation.enabled:
            write_evaluation_history(
                config.evaluation.data_path,
                evaluation_records,
                environment_id=config.environment.id,
            )
        training_progress = tqdm(
            total=total_updates,
            desc="Training",
            unit="update",
            position=0,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        self_play_progress = tqdm(
            total=config.self_play.total_transitions,
            desc="Self-play",
            unit="transition",
            position=1,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        log(
            "[bold cyan]MuZero training started[/bold cyan] "
            f"[dim]env={config.environment.id} device={device} "
            f"transitions={config.self_play.total_transitions:,} "
            f"updates={total_updates:,}[/dim]"
        )

        def checkpoint_and_evaluate() -> None:
            """Save latest/representative weights and evaluate this checkpoint."""
            if update in checkpointed_updates:
                return
            save_checkpoint(
                latest_checkpoint_path,
                agent=agent,
                trainer=trainer,
                update=update,
                config=config,
            )
            saved_path = latest_checkpoint_path
            if update in representative_updates:
                saved_path = representative_checkpoint_path(
                    latest_checkpoint_path, update
                )
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(latest_checkpoint_path, saved_path)
            checkpointed_updates.add(update)
            log(
                "[green]Checkpoint saved[/green] "
                f"[dim]update={update:,} path={saved_path}[/dim]"
            )

            if not config.evaluation.enabled:
                return
            stats = evaluate_agent(
                agent,
                lambda: create_evaluation_environment(config),
                episodes=config.evaluation.episodes,
                seed=config.seed,
            )
            evaluation_records.append(
                EvaluationRecord.create(update, saved_path, stats)
            )
            write_evaluation_history(
                config.evaluation.data_path,
                evaluation_records,
                environment_id=config.environment.id,
            )
            log(
                "[bold blue]Evaluation complete[/bold blue] "
                f"[dim]update={update:,} episodes={config.evaluation.episodes} "
                f"mean={stats.mean:.2f} median={stats.median:.2f} "
                f"std={stats.std:.2f}[/dim]"
            )

        def run_updates(count: int) -> None:
            nonlocal update
            for _ in range(count):
                priority_beta = linear_priority_beta(
                    update,
                    total_updates,
                    config.replay.priority_beta_initial,
                    config.replay.priority_beta_final,
                )
                batch = replay.sample(
                    config.training.batch_size,
                    unroll_steps=config.training.unroll_steps,
                    td_steps=config.training.td_steps,
                    discount=discount,
                    priority_beta=priority_beta,
                ).to(device)
                metrics = trainer.train_step(batch)
                replay.update_priorities(batch.indices, metrics.priorities)
                update += 1
                if update == 1 or update % config.training.log_every == 0:
                    training_progress.set_postfix(
                        loss=f"{metrics.loss:.3f}",
                        policy=f"{metrics.policy_loss:.3f}",
                        value=f"{metrics.value_loss:.3f}",
                        reward=f"{metrics.reward_loss:.3f}",
                        grad=f"{metrics.gradient_norm:.2f}",
                        lr=f"{metrics.learning_rate:.5f}",
                        beta=f"{priority_beta:.3f}",
                        refresh=False,
                    )
                training_progress.update(1)
                regular_checkpoint = (
                    config.checkpoint.every > 0
                    and update % config.checkpoint.every == 0
                )
                if regular_checkpoint or update in representative_updates:
                    checkpoint_and_evaluate()

        minimum_replay_size = max(
            config.replay.warmup_transitions,
            config.training.batch_size,
        )
        collection_iteration = 0
        with SelfPlayWorker(
            agent,
            environments=environments,
            frame_stack=config.environment.frame_stack,
            trajectory_length=config.self_play.trajectory_length,
            base_seed=config.seed,
            clip_rewards=config.self_play.clip_rewards,
            add_exploration_noise=config.self_play.add_exploration_noise,
        ) as worker:
            while worker.total_transitions < config.self_play.total_transitions:
                collection_iteration += 1
                vector_steps = next_collection_vector_steps(
                    worker.total_transitions,
                    config.self_play.total_transitions,
                    config.self_play.num_envs,
                    config.self_play.steps_per_iteration,
                )
                warming_up = len(replay) < minimum_replay_size
                temperature = visit_softmax_temperature(
                    update, config.training.steps
                )
                previous_transitions = worker.total_transitions
                grouped = worker.run(
                    vector_steps,
                    temperature=temperature,
                    random_actions=warming_up,
                )
                self_play_progress.update(
                    worker.total_transitions - previous_transitions
                )
                trajectories = tuple(flatten_trajectories(grouped))
                insertion = replay.extend(trajectories)
                self_play_progress.set_postfix(
                    iteration=collection_iteration,
                    added=insertion.added_transitions,
                    replay=f"{len(replay)}/{replay.max_transitions}",
                    temperature=f"{temperature:.2f}",
                    mode="random" if warming_up else "MCTS",
                    refresh=False,
                )
                if warming_up and len(replay) >= minimum_replay_size:
                    log(
                        "[bold green]Replay warmup complete[/bold green] "
                        f"[dim]stored={len(replay):,}[/dim]"
                    )

                if (
                    len(replay) < minimum_replay_size
                    or update >= config.training.steps
                ):
                    continue

                run_updates(
                    min(
                        config.training.updates_per_iteration,
                        config.training.steps - update,
                    )
                )

            final_trajectories = tuple(
                flatten_trajectories(worker.flush())
            )
            if final_trajectories:
                replay.extend(final_trajectories)

        self_play_progress.close()
        log(
            "[bold green]Self-play complete[/bold green] "
            f"[dim]transitions={config.self_play.total_transitions:,}[/dim]"
        )
        run_updates(config.training.steps - update)
        log(
            "[bold yellow]Final learner-only phase[/bold yellow] "
            f"[dim]updates={config.training.final_steps:,}[/dim]"
        )
        run_updates(config.training.final_steps)

        checkpoint_and_evaluate()
        if config.evaluation.enabled:
            plot_evaluation_history(
                config.evaluation.plot_path,
                evaluation_records,
                title=config.environment.id.removeprefix("ALE/").removesuffix("-v5"),
            )
            log(
                "[bold magenta]Evaluation plot saved[/bold magenta] "
                f"[dim]path={config.evaluation.plot_path}[/dim]"
            )
        training_progress.close()
        log(
            "[bold green]Training complete[/bold green] "
            f"[dim]updates={update:,} checkpoint={config.checkpoint.path}[/dim]"
        )
    except Exception:
        for progress_name in ("self_play_progress", "training_progress"):
            progress = locals().get(progress_name)
            if progress is not None:
                progress.close()
        # The worker takes ownership only after its context has been entered.
        for environment in environments:
            environment.close()
        raise


if __name__ == "__main__":
    main()
