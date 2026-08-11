#!/usr/bin/env python3
"""Train AtariAgent with MuZero losses on FIFO self-play replay."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import random
import shutil
import sys

import hydra
import numpy as np
import ray
from omegaconf import OmegaConf
from rich import print as rich_print
import torch
from tqdm.auto import tqdm

from atariagent import (
    AtariAgent,
    EpisodeRewardTracker,
    FIFOReplayBuffer,
    GameTrajectory,
    ReplayBatch,
    SelfPlayWorker,
)
from atariagent.evaluation import (
    EvaluationRecord,
    EvaluationStats,
    evaluate_agent,
    plot_evaluation_history,
    write_evaluation_history,
)
from atariagent.search import MCTS, MCTSConfig
from atariagent.selfplay import Environment, make_atari_environment
from atariagent.training import (
    MuZeroTrainer,
    ReanalysisPipeline,
    create_reanalysis_actors,
    initialize_local_ray,
    make_target_state,
    representative_checkpoint_path,
    representative_checkpoint_updates,
)
from atariagent.typecheck import set_runtime_typechecking
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


def configure_training_backend(deterministic: bool) -> None:
    """Select reproducible kernels or faster cuDNN/TF32 execution."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.set_float32_matmul_precision("highest" if deterministic else "high")


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
    target_state: Mapping[str, torch.Tensor],
    target_version: int,
    update: int,
    config: TrainMuZeroConfig,
) -> None:
    """Persist online networks, asynchronous target state, and optimizer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, object] = {
        "update": update,
        "representation": agent.representation_network.state_dict(),
        "dynamics": agent.dynamics_network.state_dict(),
        "prediction": agent.prediction_network.state_dict(),
        "target_network": dict(target_state),
        "target_version": target_version,
        "optimizer": trainer.optimizer.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    torch.save(checkpoint, path)


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
    if config.training.reanalysis_start_step < 0:
        raise ValueError("reanalysis_start_step must be non-negative")
    if config.training.target_update_interval <= 0:
        raise ValueError("target_update_interval must be positive")
    if not 0.0 <= config.training.policy_reanalysis_ratio <= 1.0:
        raise ValueError("policy_reanalysis_ratio must be in [0, 1]")
    if config.training.policy_reanalysis_chunk_size <= 0:
        raise ValueError("policy_reanalysis_chunk_size must be positive")
    if config.training.reanalysis_actor_count <= 0:
        raise ValueError("reanalysis_actor_count must be positive")
    if config.training.reanalysis_actor_num_threads <= 0:
        raise ValueError("reanalysis_actor_num_threads must be positive")
    if (
        config.training.reanalysis_prefetch_batches
        < config.training.reanalysis_actor_count
    ):
        raise ValueError(
            "reanalysis_prefetch_batches must cover all reanalysis actors"
        )
    if config.training.precision not in ("fp32", "bf16"):
        raise ValueError("training.precision must be fp32 or bf16")
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
    configure_training_backend(config.training.deterministic)
    set_runtime_typechecking(config.training.runtime_type_checks)
    device = resolve_device(config.training.device)
    environments: list[Environment] = []
    reanalysis_pipeline: ReanalysisPipeline | None = None
    owns_ray = False

    try:
        owns_ray = initialize_local_ray(
            object_store_memory=config.training.ray_object_store_memory
        )
        environments = create_environments(config)
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
            precision=config.training.precision,
            compile_model=config.training.compile_model,
            compile_mode=config.training.compile_mode,
        )
        replay = FIFOReplayBuffer(
            config.replay.max_transitions,
            seed=config.seed,
            priority_alpha=config.replay.priority_alpha,
        )
        policy_reanalysis_enabled = (
            config.training.policy_reanalysis_ratio > 0.0
        )
        actors = create_reanalysis_actors(
            count=config.training.reanalysis_actor_count,
            num_gpus=(
                config.training.reanalysis_actor_num_gpus
                if torch.cuda.is_available()
                else 0.0
            ),
            num_cpus=config.training.reanalysis_actor_num_threads,
            mcts_threads=config.training.reanalysis_actor_num_threads,
            in_channels=config.environment.frame_stack * image_channels,
            action_space_size=action_space_size,
            mcts_config=agent.mcts.config,
            policy_enabled=policy_reanalysis_enabled,
            rng_seed=config.seed,
            support_min=-300,
            support_max=300,
            precision=config.training.precision,
        )
        reanalysis_pipeline = ReanalysisPipeline(
            actors,
            reanalyze_values=config.training.use_target_network_reanalysis,
            policy_ratio=config.training.policy_reanalysis_ratio,
            policy_chunk_size=(
                config.training.policy_reanalysis_chunk_size
            ),
            prefetch_batches=(
                config.training.reanalysis_prefetch_batches
            ),
            timeout_seconds=config.training.reanalysis_timeout_seconds,
            max_weight_lag=config.training.reanalysis_max_weight_lag,
        )
        initial_target_state = make_target_state(
            agent.representation_network,
            agent.prediction_network,
            (
                agent.dynamics_network
                if policy_reanalysis_enabled
                else None
            ),
        )
        reanalysis_pipeline.publish_weights(
            0,
            initial_target_state,
            wait=True,
        )

        total_updates = config.training.steps + config.training.final_steps
        representative_updates = set(
            representative_checkpoint_updates(
                total_updates, config.checkpoint.keep_representative
            )
        )
        evaluation_records: list[EvaluationRecord] = []
        checkpointed_updates: set[int] = set()
        reward_tracker = EpisodeRewardTracker()
        self_play_episode_rewards: list[float] = []
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
            f"precision={config.training.precision} "
            f"deterministic={config.training.deterministic} "
            f"compile={config.training.compile_model} "
            f"transitions={config.self_play.total_transitions:,} "
            f"updates={total_updates:,}[/dim]"
        )

        def checkpoint_and_evaluate() -> None:
            """Save scheduled weights and evaluate representative checkpoints."""
            if update in checkpointed_updates:
                return
            is_representative = update in representative_updates
            assert reanalysis_pipeline is not None
            target_state = reanalysis_pipeline.latest_target_state
            if target_state is None:
                raise RuntimeError("asynchronous target state is unavailable")
            save_checkpoint(
                latest_checkpoint_path,
                agent=agent,
                trainer=trainer,
                target_state=target_state,
                target_version=reanalysis_pipeline.weight_version,
                update=update,
                config=config,
            )
            saved_path = latest_checkpoint_path
            if is_representative:
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

            if not config.evaluation.enabled or not is_representative:
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

        def sample_batch(
            sample_step: int,
            *,
            include_value_bootstraps: bool,
        ) -> tuple[ReplayBatch, float]:
            priority_beta = linear_priority_beta(
                sample_step,
                total_updates,
                config.replay.priority_beta_initial,
                config.replay.priority_beta_final,
            )
            return (
                replay.sample(
                    config.training.batch_size,
                    unroll_steps=config.training.unroll_steps,
                    td_steps=config.training.td_steps,
                    discount=discount,
                    priority_beta=priority_beta,
                    include_value_bootstraps=include_value_bootstraps,
                ),
                priority_beta,
            )

        def apply_update(
            cpu_batch: ReplayBatch,
            priority_beta: float,
            *,
            queue_wait_ms: float | None = None,
            actor_duration_ms: float | None = None,
        ) -> None:
            nonlocal update
            assert reanalysis_pipeline is not None
            pin_batch = config.training.pin_memory and device.type == "cuda"
            if pin_batch:
                cpu_batch = cpu_batch.pin_memory()
            batch = cpu_batch.to(
                device,
                non_blocking=pin_batch,
                keep_indices_on_cpu=True,
            )
            metrics = trainer.train_step(batch)
            replay.update_priorities(batch.indices, metrics.priorities)

            update += 1
            if update % config.training.target_update_interval == 0:
                target_state = make_target_state(
                    agent.representation_network,
                    agent.prediction_network,
                    (
                        agent.dynamics_network
                        if policy_reanalysis_enabled
                        else None
                    ),
                )
                reanalysis_pipeline.publish_weights(update, target_state)

            if update == 1 or update % config.training.log_every == 0:
                progress_stats: dict[str, str] = {
                    "loss": f"{metrics.loss:.3f}",
                    "policy": f"{metrics.policy_loss:.3f}",
                    "value": f"{metrics.value_loss:.3f}",
                    "reward": f"{metrics.reward_loss:.3f}",
                    "grad": f"{metrics.gradient_norm:.2f}",
                    "lr": f"{metrics.learning_rate:.5f}",
                    "beta": f"{priority_beta:.3f}",
                }
                if queue_wait_ms is not None and actor_duration_ms is not None:
                    progress_stats.update(
                        {
                            "reanalyze": f"{actor_duration_ms:.0f}ms",
                            "queue": f"{queue_wait_ms:.0f}ms",
                            "pending": str(reanalysis_pipeline.pending_count),
                        }
                    )
                training_progress.set_postfix(progress_stats, refresh=False)
            training_progress.update(1)

            regular_checkpoint = (
                config.checkpoint.every > 0
                and update % config.checkpoint.every == 0
            )
            if regular_checkpoint or update in representative_updates:
                checkpoint_and_evaluate()

        def run_updates(count: int) -> None:
            if count <= 0:
                return
            assert reanalysis_pipeline is not None

            direct_updates = min(
                count,
                max(0, config.training.reanalysis_start_step - update),
            )
            for _ in range(direct_updates):
                cpu_batch, priority_beta = sample_batch(
                    update,
                    include_value_bootstraps=False,
                )
                apply_update(cpu_batch, priority_beta)

            async_updates = count - direct_updates
            if async_updates <= 0:
                return
            async_start_step = update
            submitted = 0
            completed = 0
            priority_betas: dict[int, float] = {}
            while completed < async_updates:
                while (
                    submitted < async_updates
                    and reanalysis_pipeline.needs_prefetch
                ):
                    cpu_batch, priority_beta = sample_batch(
                        async_start_step + submitted,
                        include_value_bootstraps=(
                            config.training.use_target_network_reanalysis
                        ),
                    )
                    request_id = reanalysis_pipeline.submit(cpu_batch)
                    priority_betas[request_id] = priority_beta
                    submitted += 1

                ready = reanalysis_pipeline.wait_next()
                priority_beta = priority_betas.pop(ready.request_id)
                apply_update(
                    ready.batch,
                    priority_beta,
                    queue_wait_ms=ready.queue_wait_ms,
                    actor_duration_ms=ready.actor_duration_ms,
                )
                completed += 1

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
                # Temperature affects only rollout behavior/action selection;
                # training targets always normalize the raw MCTS visit counts.
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
                completed_rewards = reward_tracker.add(trajectories)
                self_play_episode_rewards.extend(completed_rewards)
                insertion = replay.extend(trajectories)
                progress_stats: dict[str, object] = {
                    "iteration": collection_iteration,
                    "added": insertion.added_transitions,
                    "replay": f"{len(replay)}/{replay.max_transitions}",
                    "temperature": f"{temperature:.2f}",
                    "mode": "random" if warming_up else "MCTS",
                }
                if self_play_episode_rewards:
                    recent_stats = EvaluationStats.from_rewards(
                        tuple(self_play_episode_rewards[-100:])
                    )
                    progress_stats["full_episodes"] = len(
                        self_play_episode_rewards
                    )
                    progress_stats["full_game_reward100"] = (
                        f"{recent_stats.mean:.2f}"
                    )
                self_play_progress.set_postfix(
                    progress_stats,
                    refresh=False,
                )
                if completed_rewards:
                    recent_rewards = self_play_episode_rewards[-100:]
                    recent_stats = EvaluationStats.from_rewards(
                        tuple(recent_rewards)
                    )
                    log(
                        "[bold cyan]Self-play reward statistics"
                        "[/bold cyan] "
                        f"[dim]update={update:,} iteration={collection_iteration} "
                        f"new_episodes={len(completed_rewards)} "
                        f"total_episodes={len(self_play_episode_rewards):,} "
                        f"window={len(recent_rewards)} "
                        f"mean={recent_stats.mean:.2f} "
                        f"median={recent_stats.median:.2f} "
                        f"std={recent_stats.std:.2f} "
                        f"min={min(recent_rewards):.2f} "
                        f"max={max(recent_rewards):.2f} "
                        f"latest={completed_rewards[-1]:.2f} "
                        f"mode={'random' if warming_up else 'MCTS'}[/dim]"
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
        if self_play_episode_rewards:
            recent_rewards = self_play_episode_rewards[-100:]
            recent_stats = EvaluationStats.from_rewards(tuple(recent_rewards))
            reward_summary = (
                f"full_episodes={len(self_play_episode_rewards):,} "
                f"full_game_reward100_mean={recent_stats.mean:.2f} "
                f"full_game_reward100_median={recent_stats.median:.2f} "
                f"full_game_reward100_std={recent_stats.std:.2f}"
            )
        else:
            reward_summary = "full_episodes=0"
        log(
            "[bold green]Self-play complete[/bold green] "
            f"[dim]transitions={config.self_play.total_transitions:,} "
            f"{reward_summary}[/dim]"
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
    finally:
        if reanalysis_pipeline is not None:
            try:
                reanalysis_pipeline.close()
            except Exception:
                pass
        if owns_ray and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
