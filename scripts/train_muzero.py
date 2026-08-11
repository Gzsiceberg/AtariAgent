#!/usr/bin/env python3
"""Train AtariAgent with MuZero losses on FIFO self-play replay."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter

import hydra
import numpy as np
from omegaconf import OmegaConf
from rich import print as rich_print
import torch
from tqdm.auto import tqdm

from atariagent import (
    AtariAgent,
    EpisodeRewardTracker,
    FIFOReplayBuffer,
    GameTrajectory,
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
    LearnerProfiler,
    MuZeroTrainer,
    representative_checkpoint_path,
    representative_checkpoint_updates,
)
from atariagent.training.profiling import synchronize_for_profiling
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
    update: int,
    config: TrainMuZeroConfig,
) -> None:
    """Persist online networks, optional target network, and optimizer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, object] = {
        "update": update,
        "representation": agent.representation_network.state_dict(),
        "dynamics": agent.dynamics_network.state_dict(),
        "prediction": agent.prediction_network.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    if trainer.target_network is not None:
        checkpoint["target_network"] = trainer.target_network.state_dict()
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
    if config.training.precision not in ("fp32", "bf16"):
        raise ValueError("training.precision must be fp32 or bf16")
    if config.training.profile_warmup_steps < 0:
        raise ValueError("profile_warmup_steps must be non-negative")
    if config.training.profile_report_every <= 0:
        raise ValueError("profile_report_every must be positive")
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
            use_target_network_reanalysis=(
                config.training.use_target_network_reanalysis
            ),
            policy_reanalysis_ratio=(
                config.training.policy_reanalysis_ratio
            ),
            policy_reanalysis_chunk_size=(
                config.training.policy_reanalysis_chunk_size
            ),
            action_space_size=action_space_size,
            mcts_config=agent.mcts.config,
            reanalysis_seed=config.seed,
            target_update_interval=config.training.target_update_interval,
            precision=config.training.precision,
            compile_model=config.training.compile_model,
            compile_mode=config.training.compile_mode,
        )
        replay = FIFOReplayBuffer(
            config.replay.max_transitions,
            seed=config.seed,
            priority_alpha=config.replay.priority_alpha,
        )

        total_updates = config.training.steps + config.training.final_steps
        learner_profiler = (
            LearnerProfiler(
                warmup_steps=config.training.profile_warmup_steps,
                report_every=config.training.profile_report_every,
                batch_size=config.training.batch_size,
                device=device,
            )
            if config.training.profile
            else None
        )
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
            save_checkpoint(
                latest_checkpoint_path,
                agent=agent,
                trainer=trainer,
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

        def run_updates(count: int) -> None:
            nonlocal update
            if count <= 0:
                return

            profiling = config.training.profile
            pin_batches = config.training.pin_memory and device.type == "cuda"
            for _ in range(count):
                synchronize_for_profiling(device, profiling)
                update_started = perf_counter() if profiling else 0.0
                timings_ms: dict[str, float] = {}

                priority_beta = linear_priority_beta(
                    update,
                    total_updates,
                    config.replay.priority_beta_initial,
                    config.replay.priority_beta_final,
                )
                sampling_started = perf_counter() if profiling else 0.0
                cpu_batch = replay.sample(
                    config.training.batch_size,
                    unroll_steps=config.training.unroll_steps,
                    td_steps=config.training.td_steps,
                    discount=discount,
                    priority_beta=priority_beta,
                    include_value_bootstraps=(
                        config.training.use_target_network_reanalysis
                    ),
                )
                if profiling:
                    timings_ms["replay_sample"] = (
                        perf_counter() - sampling_started
                    ) * 1_000.0

                if pin_batches:
                    cpu_batch = cpu_batch.pin_memory()
                batch = cpu_batch.to(
                    device,
                    non_blocking=pin_batches,
                    keep_indices_on_cpu=True,
                )

                metrics = trainer.train_step(batch, profile=profiling)
                if metrics.timings_ms is not None:
                    timings_ms.update(metrics.timings_ms)

                priority_started = perf_counter() if profiling else 0.0
                replay.update_priorities(batch.indices, metrics.priorities)
                if profiling:
                    timings_ms["priority_update"] = (
                        perf_counter() - priority_started
                    ) * 1_000.0
                    synchronize_for_profiling(device, True)
                    timings_ms["update_total"] = (
                        perf_counter() - update_started
                    ) * 1_000.0

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

                if learner_profiler is not None:
                    summary = learner_profiler.record(timings_ms)
                    if summary is not None:
                        sections = " ".join(
                            f"{name}={duration:.2f}ms"
                            for name, duration in sorted(
                                summary.mean_sections_ms.items()
                            )
                            if name != "update_total"
                        )
                        log(
                            "[bold magenta]Learner profile[/bold magenta] "
                            f"[dim]updates={summary.updates} "
                            f"ups={summary.updates_per_second:.2f} "
                            f"samples/s={summary.samples_per_second:.0f} "
                            f"update_mean={summary.mean_update_ms:.2f}ms "
                            f"update_p95={summary.p95_update_ms:.2f}ms "
                            f"peak_alloc={summary.peak_allocated_mib:.0f}MiB "
                            f"peak_reserved={summary.peak_reserved_mib:.0f}MiB "
                            f"{sections}[/dim]"
                        )

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


if __name__ == "__main__":
    main()
