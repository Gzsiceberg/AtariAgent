#!/usr/bin/env python3
"""Train AtariAgent from FIFO self-play replay."""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from rich import print as rich_print
from rich.text import Text
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
from atariagent.models import ConsistencyNetwork
from atariagent.search import (
    SearchConfig,
    efficientzero_atari_gumbel_settings,
)
from atariagent.selfplay import Environment, make_atari_environment
from atariagent.training import (
    BatchWorker,
    ReadyBatch,
    ReanalysisPipeline,
    Trainer,
    WandbLogger,
    make_target_state,
    representative_checkpoint_path,
    representative_checkpoint_updates,
    wandb_run_name,
)
from atariagent.training.config import (
    TrainAgentConfig,
    final_evaluation_max_episode_steps,
    next_collection_vector_steps,
    puct_root_noise_temperature,
    register_train_agent_config,
    scheduled_target_update_interval,
    target_network_update_due,
    visit_softmax_temperature,
)
from atariagent.typecheck import set_runtime_typechecking

register_train_agent_config()


def log(message: str) -> None:
    """Print one plain line when redirected, or Rich markup interactively."""
    if not sys.stdout.isatty():
        print(Text.from_markup(message).plain, flush=True)
        return
    with tqdm.external_write_mode():
        rich_print(message)


def repository_commit() -> str:
    """Return the commit checked out in this script's repository."""
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve_device(name: str) -> torch.device:
    """Resolve the required CUDA training device."""
    if not torch.cuda.is_available():
        raise RuntimeError("AtariAgent training requires a CUDA GPU")
    device = torch.device("cuda" if name == "auto" else name)
    if device.type != "cuda":
        raise ValueError("training.device must select a CUDA GPU")
    return device


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


def create_environments(config: TrainAgentConfig) -> list[Environment]:
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


def create_evaluation_environment(config: TrainAgentConfig) -> Environment:
    """Create a full-episode Atari environment for policy evaluation."""
    return make_atari_environment(
        config.environment.id,
        frame_stack=config.environment.frame_stack,
        frame_skip=config.environment.frame_skip,
        screen_size=config.environment.screen_size,
        max_episode_steps=final_evaluation_max_episode_steps(
            config.environment.frame_skip
        ),
        grayscale_obs=config.environment.grayscale,
        terminal_on_life_loss=False,
    )


def save_checkpoint(
    path: Path,
    *,
    agent: AtariAgent,
    trainer: Trainer,
    target_state: Mapping[str, torch.Tensor],
    target_version: int,
    update: int,
    config: TrainAgentConfig,
) -> None:
    """Persist online networks, asynchronous target state, and optimizer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, object] = {
        "update": update,
        "representation": agent.representation_network.state_dict(),
        "dynamics": agent.dynamics_network.state_dict(),
        "prediction": agent.prediction_network.state_dict(),
        "consistency": (
            trainer.consistency_network.state_dict()
            if trainer.consistency_network is not None
            else None
        ),
        "target_network": dict(target_state),
        "target_version": target_version,
        "optimizer": trainer.optimizer.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    torch.save(checkpoint, path)


@hydra.main(version_base=None, config_path="../configs", config_name="train_agent")
def main(config: TrainAgentConfig) -> None:
    """Alternate self-play collection with updates sampled from replay."""
    start_time = datetime.now().astimezone().isoformat(timespec="seconds")
    commit = repository_commit()
    log(
        "[bold cyan]AtariAgent launch[/bold cyan] "
        f"[dim]commit={commit} start_time={start_time}[/dim]"
    )
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
    if config.training.value_target not in {"td", "search", "mixed"}:
        raise ValueError("training.value_target must be td, search, or mixed")
    for value, name in (
        (
            config.training.mixed_value_start_step,
            "mixed_value_start_step",
        ),
        (
            config.training.mixed_value_threshold,
            "mixed_value_threshold",
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"training.{name} must be an integer")
        if value < 0:
            raise ValueError(f"training.{name} must be non-negative")
    if not isinstance(config.reanalysis.enabled, bool):
        raise TypeError("reanalysis.enabled must be a boolean")
    if (
        config.training.value_target in {"search", "mixed"}
        and not config.reanalysis.enabled
    ):
        raise ValueError("search value targets require target reanalysis")
    for value, name in (
        (
            config.reanalysis.initial_cache_clear_interval,
            "initial_cache_clear_interval",
        ),
        (
            config.reanalysis.initial_target_update_interval,
            "initial_target_update_interval",
        ),
        (config.reanalysis.target_update_interval, "target_update_interval"),
        (
            config.reanalysis.target_update_ramp_steps,
            "target_update_ramp_steps",
        ),
        (config.reanalysis.target_update_stages, "target_update_stages"),
        (
            config.reanalysis.policy_reanalysis_ramp_transitions,
            "policy_reanalysis_ramp_transitions",
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"reanalysis.{name} must be an integer")
        if value <= 0:
            raise ValueError(f"reanalysis.{name} must be positive")
    for initial_interval, name in (
        (
            config.reanalysis.initial_cache_clear_interval,
            "initial_cache_clear_interval",
        ),
        (
            config.reanalysis.initial_target_update_interval,
            "initial_target_update_interval",
        ),
    ):
        if initial_interval > config.reanalysis.target_update_interval:
            raise ValueError(
                f"reanalysis.{name} must not exceed "
                "reanalysis.target_update_interval"
            )
    if config.reanalysis.target_update_stages < 2:
        raise ValueError("reanalysis.target_update_stages must be at least two")
    if config.reanalysis.policy_chunk_size <= 0:
        raise ValueError("reanalysis.policy_chunk_size must be positive")
    if not isinstance(config.reanalysis.cache_targets, bool):
        raise TypeError("reanalysis.cache_targets must be a boolean")
    if isinstance(config.reanalysis.cache_target_ttl, bool) or not isinstance(
        config.reanalysis.cache_target_ttl, int
    ):
        raise TypeError("reanalysis.cache_target_ttl must be an integer")
    if config.reanalysis.cache_target_ttl < 0:
        raise ValueError("reanalysis.cache_target_ttl must be non-negative")
    if config.reanalysis.worker_num_threads <= 0:
        raise ValueError("reanalysis.worker_num_threads must be positive")
    if config.reanalysis.prefetch_batches <= 0:
        raise ValueError("reanalysis.prefetch_batches must be positive")
    if config.training.batch_max_in_flight <= 0:
        raise ValueError("batch_max_in_flight must be positive")
    if config.training.batch_ready_prefetch <= 0:
        raise ValueError("batch_ready_prefetch must be positive")
    if (
        config.training.batch_ready_prefetch
        > config.training.batch_max_in_flight
    ):
        raise ValueError(
            "batch_ready_prefetch must not exceed batch_max_in_flight"
        )
    if config.training.batch_worker_timeout_seconds <= 0.0:
        raise ValueError("batch_worker_timeout_seconds must be positive")
    if config.training.precision not in ("fp32", "bf16"):
        raise ValueError("training.precision must be fp32 or bf16")
    if not isinstance(config.augmentation.enabled, bool):
        raise TypeError("augmentation.enabled must be a boolean")
    if config.augmentation.shift_delta < 0:
        raise ValueError("augmentation.shift_delta must be non-negative")
    if config.augmentation.intensity_scale < 0.0:
        raise ValueError("augmentation.intensity_scale must be non-negative")
    if config.augmentation.enabled and not config.augmentation.transforms:
        raise ValueError("augmentation.transforms must not be empty when enabled")
    if not isinstance(config.loss.consistency_enabled, bool):
        raise TypeError("loss.consistency_enabled must be a boolean")
    if config.loss.consistency_weight < 0.0:
        raise ValueError("loss.consistency_weight must be non-negative")
    if config.checkpoint.keep_representative <= 0:
        raise ValueError("checkpoint.keep_representative must be positive")
    if config.evaluation.enabled and config.evaluation.episodes <= 0:
        raise ValueError("evaluation.episodes must be positive")
    if config.evaluation.enabled and config.evaluation.num_envs <= 0:
        raise ValueError("evaluation.num_envs must be positive")
    if not isinstance(config.wandb.enabled, bool):
        raise TypeError("wandb.enabled must be a boolean")
    if config.wandb.enabled and not config.wandb.project.strip():
        raise ValueError("wandb.project must not be empty when enabled")
    if config.self_play.num_envs <= 0:
        raise ValueError("self_play.num_envs must be positive")
    if config.self_play.search_algorithm not in {"puct", "mcts", "gumbel"}:
        raise ValueError("self_play.search_algorithm must be puct or gumbel")
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
    batch_worker: BatchWorker | None = None
    wandb_logger = WandbLogger()
    wandb_exit_code = 1

    try:
        target_reanalysis_enabled = config.reanalysis.enabled
        environments = create_environments(config)
        action_space_size = int(environments[0].action_space.n)
        if any(
            int(environment.action_space.n) != action_space_size
            for environment in environments
        ):
            raise ValueError("all environments must have the same action space")

        num_top_actions = 4  # Unused by PUCT.
        if config.self_play.search_algorithm == "gumbel":
            (
                config.self_play.num_simulations,
                num_top_actions,
            ) = efficientzero_atari_gumbel_settings(
                action_space_size,
                config.self_play.num_simulations,
            )
            print(
                f"num_simulations={config.self_play.num_simulations}, "
                f"num_top_actions={num_top_actions}, "
                f"action_space_size={action_space_size}"
            )

        resolved_config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(resolved_config, Mapping):
            raise TypeError("resolved training config must be a mapping")
        wandb_logger = WandbLogger.initialize(
            config.wandb,
            run_name=wandb_run_name(config.environment.id, commit),
            run_config=resolved_config,
        )

        image_channels = 1 if config.environment.grayscale else 3
        discount = config.training.discount ** config.environment.frame_skip
        agent = AtariAgent(
            config.environment.frame_stack * image_channels,
            action_space_size,
            search_config=SearchConfig(
                num_simulations=config.self_play.num_simulations,
                discount=discount,
                value_prefix_horizon=config.training.lstm_horizon,
                search_algorithm=config.self_play.search_algorithm,
                num_top_actions=num_top_actions,
                c_visit=config.self_play.c_visit,
                c_scale=config.self_play.c_scale,
            ),
            search_rng=random.Random(config.seed),
        ).to(device)
        consistency_network = (
            ConsistencyNetwork().to(device)
            if config.loss.consistency_enabled
            else None
        )
        trainer = Trainer(
            agent.representation_network,
            agent.dynamics_network,
            agent.prediction_network,
            consistency_network=consistency_network,
            augmentation=(
                config.augmentation.transforms
                if config.augmentation.enabled
                else None
            ),
            augmentation_shift_delta=config.augmentation.shift_delta,
            augmentation_intensity_scale=(
                config.augmentation.intensity_scale
            ),
            image_shape=(
                config.environment.screen_size,
                config.environment.screen_size,
            ),
            optimizer=config.training.optimizer,
            learning_rate=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
            lr_warmup_steps=config.training.lr_warmup_steps,
            lr_decay_rate=config.training.lr_decay_rate,
            lr_decay_steps=config.training.lr_decay_steps,
            steps=config.training.steps,
            final_steps=config.training.final_steps,
            unroll_steps=config.training.unroll_steps,
            lstm_horizon=config.training.lstm_horizon,
            policy_weight=config.loss.policy_weight,
            value_weight=config.loss.value_weight,
            reward_weight=config.loss.reward_weight,
            consistency_weight=config.loss.consistency_weight,
            max_gradient_norm=config.training.max_gradient_norm,
            priority_epsilon=config.replay.priority_epsilon,
            precision=config.training.precision,
            compile_model=config.training.compile_model,
            compile_mode=config.training.compile_mode,
        )
        replay = FIFOReplayBuffer(
            config.replay.max_transitions,
            unroll_steps=config.training.unroll_steps,
            td_steps=config.training.td_steps,
            discount=discount,
            priority_epsilon=config.replay.priority_epsilon,
            seed=config.seed,
        )
        target_state = make_target_state(
            agent.representation_network,
            agent.prediction_network,
            (
                agent.dynamics_network
                if target_reanalysis_enabled
                else None
            ),
        )
        target_version = 0
        if target_reanalysis_enabled:
            reanalysis_pipeline = ReanalysisPipeline(
                in_channels=config.environment.frame_stack * image_channels,
                action_space_size=action_space_size,
                search_config=agent.search.config,
                policy_chunk_size=config.reanalysis.policy_chunk_size,
                cache_targets=config.reanalysis.cache_targets,
                cache_target_ttl=config.reanalysis.cache_target_ttl,
                policy_reanalysis_ramp_transitions=(
                    config.reanalysis.policy_reanalysis_ramp_transitions
                ),
                rng_seed=config.seed,
                support_min=-300,
                support_max=300,
                precision=config.training.precision,
                search_threads=config.reanalysis.worker_num_threads,
                prefetch_batches=config.reanalysis.prefetch_batches,
                timeout_seconds=config.reanalysis.timeout_seconds,
                target_update_interval=(
                    config.reanalysis.target_update_interval
                ),
                root_noise_total_steps=config.training.steps,
                device=device,
            )
            reanalysis_pipeline.publish_weights(
                target_version,
                target_state,
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
            disable=not sys.stdout.isatty(),
        )
        self_play_progress = tqdm(
            total=config.self_play.total_transitions,
            desc="Self-play",
            unit="transition",
            position=1,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not sys.stdout.isatty(),
        )
        log(
            "[bold cyan]AtariAgent training started[/bold cyan] "
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
                target_state=target_state,
                target_version=target_version,
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
                num_envs=config.evaluation.num_envs,
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
            wandb_logger.log_evaluation(stats, update=update)
            log(
                "[bold blue]Evaluation complete[/bold blue] "
                f"[dim]update={update:,} episodes={config.evaluation.episodes} "
                f"mean={stats.mean:.2f} median={stats.median:.2f} "
                f"std={stats.std:.2f} max={max(stats.rewards):.2f}[/dim]"
            )

        batch_worker = BatchWorker(
            replay,
            batch_size=config.training.batch_size,
            device=device,
            reanalysis_pipeline=reanalysis_pipeline,
            value_target=config.training.value_target,
            mixed_value_start_step=(
                config.training.mixed_value_start_step
            ),
            mixed_value_threshold=config.training.mixed_value_threshold,
            reanalysis_initial_cache_clear_interval=(
                config.reanalysis.initial_cache_clear_interval
            ),
            reanalysis_final_cache_clear_interval=(
                config.reanalysis.target_update_interval
            ),
            reanalysis_cache_clear_ramp_steps=max(
                config.training.steps // 2,
                1,
            ),
            max_in_flight=config.training.batch_max_in_flight,
            ready_prefetch=config.training.batch_ready_prefetch,
            timeout_seconds=config.training.batch_worker_timeout_seconds,
        )

        def apply_gpu_update(ready: ReadyBatch) -> None:
            nonlocal update, target_state, target_version
            metrics = trainer.train_step(ready.gpu_batch)
            # Priority transfer synchronizes the learner stream, so it also
            # makes the pinned H2D source safe to release immediately.
            batch_worker.complete(ready, metrics.priorities)

            update += 1
            target_update_interval = scheduled_target_update_interval(
                update,
                ramp_steps=config.reanalysis.target_update_ramp_steps,
                initial_interval=(
                    config.reanalysis.initial_target_update_interval
                ),
                final_interval=config.reanalysis.target_update_interval,
                stages=config.reanalysis.target_update_stages,
            )
            if target_network_update_due(
                update,
                last_update=target_version,
                interval=target_update_interval,
            ):
                target_state = make_target_state(
                    agent.representation_network,
                    agent.prediction_network,
                    (
                        agent.dynamics_network
                        if target_reanalysis_enabled
                        else None
                    ),
                )
                target_version = update
                batch_worker.publish_weights(update, target_state)

            if update == 1 or update % config.training.log_every == 0:
                wandb_logger.log_training(
                    metrics,
                    update=update,
                    policy_roots_requested=ready.policy_roots_requested,
                    policy_roots_searched=ready.policy_roots_searched,
                    cache_hits=ready.cache_hits,
                    cache_target_age_mean=ready.cache_target_age_mean,
                    cache_target_age_max=ready.cache_target_age_max,
                )
                progress_stats: dict[str, str] = {
                    # "loss": f"{metrics.loss:.3f}",
                    # "policy": f"{metrics.policy_loss:.3f}",
                    # "value": f"{metrics.value_loss:.3f}",
                    # "reward": f"{metrics.reward_loss:.3f}",
                    # "consistency": f"{metrics.consistency_loss:.3f}",
                    # "grad": f"{metrics.gradient_norm:.2f}",
                    "lr": f"{metrics.learning_rate:.5f}",
                }
                if (
                    ready.queue_wait_ms is not None
                    and ready.worker_duration_ms is not None
                ):
                    progress_stats.update(
                        {
                            "reanalyze": f"{ready.worker_duration_ms:.0f}ms",
                            "queue": f"{ready.queue_wait_ms:.0f}ms",
                        }
                    )
                    if ready.policy_roots_requested > 0:
                        progress_stats.update(
                            {
                                "roots": (
                                    f"{ready.policy_roots_searched}/"
                                    f"{ready.policy_roots_requested}"
                                ),
                                "hit": (
                                    f"{ready.cache_hits / ready.policy_roots_requested:.0%}"
                                ),
                                "age": (
                                    f"{ready.cache_target_age_mean:.0f}/"
                                    f"{ready.cache_target_age_max}"
                                ),
                                "cache": str(batch_worker.cache_size),
                            }
                        )
                training_progress.set_postfix(progress_stats, refresh=False)
            training_progress.update(1)

            if update in representative_updates:
                checkpoint_and_evaluate()

        def run_updates(count: int) -> None:
            if count <= 0:
                return
            batch_worker.start(update, count)
            for _ in range(count):
                ready = batch_worker.next_ready()
                ready.wait_for_current_stream(device)
                apply_gpu_update(ready)
            batch_worker.wait_idle()

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
            lookahead_steps=max(
                config.training.unroll_steps,
                config.training.td_steps,
            ),
            base_seed=config.seed,
            clip_rewards=config.self_play.clip_rewards,
        ) as worker:
            search_mode = agent.search.config.search_algorithm.upper()
            while worker.total_transitions < config.self_play.total_transitions:
                collection_iteration += 1
                vector_steps = next_collection_vector_steps(
                    worker.total_transitions,
                    config.self_play.total_transitions,
                    config.self_play.num_envs,
                    config.self_play.steps_per_iteration,
                )
                warming_up = len(replay) < minimum_replay_size
                collection_mode = "RANDOM" if warming_up else search_mode
                previous_transitions = worker.total_transitions
                if agent.search.config.search_algorithm == "puct":
                    temperature = visit_softmax_temperature(
                        update, config.training.steps
                    )
                    root_noise_temperature = puct_root_noise_temperature(
                        update, config.training.steps
                    )
                    grouped = worker.run(
                        vector_steps,
                        temperature=temperature,
                        root_noise_temperature=root_noise_temperature,
                        gumbel_sampling=False,
                        random_policy=warming_up,
                    )
                else:
                    temperature = None
                    root_noise_temperature = None
                    grouped = worker.run(
                        vector_steps,
                        root_noise_temperature=0.0,
                        gumbel_sampling=True,
                        random_policy=warming_up,
                    )
                self_play_progress.update(
                    worker.total_transitions - previous_transitions
                )
                if worker.last_behavior_metrics is not None:
                    wandb_logger.log_behavior_policy(
                        worker.last_behavior_metrics,
                        total_transitions=worker.total_transitions,
                        update=update,
                    )
                trajectories = tuple(flatten_trajectories(grouped))
                completed_rewards = reward_tracker.add(trajectories)
                self_play_episode_rewards.extend(completed_rewards)
                insertion = replay.extend(trajectories)
                progress_stats: dict[str, object] = {
                    "iteration": collection_iteration,
                    "added": insertion.added_transitions,
                    "replay": f"{len(replay)}/{replay.max_transitions}",
                    "mode": collection_mode,
                }
                if temperature is not None and not warming_up:
                    progress_stats["temperature"] = f"{temperature:.2f}"
                if root_noise_temperature is not None and not warming_up:
                    progress_stats["root_noise"] = (
                        f"{root_noise_temperature:.2f}"
                    )
                if self_play_episode_rewards:
                    recent_stats = EvaluationStats.from_rewards(
                        tuple(self_play_episode_rewards[-10:])
                    )
                    progress_stats["full_episodes"] = len(
                        self_play_episode_rewards
                    )
                    progress_stats["reward_mean_10"] = (
                        f"{recent_stats.mean:.2f}"
                    )
                    progress_stats["reward_min_10"] = (
                        f"{min(recent_stats.rewards):.2f}"
                    )
                self_play_progress.set_postfix(
                    progress_stats,
                    refresh=False,
                )
                if completed_rewards:
                    recent_rewards = self_play_episode_rewards[-10:]
                    recent_stats = EvaluationStats.from_rewards(
                        tuple(recent_rewards)
                    )
                    wandb_logger.log_self_play(
                        recent_stats,
                        recent_rewards=recent_rewards,
                        total_episodes=len(self_play_episode_rewards),
                        update=update,
                    )
                    log(
                        "[bold cyan]Self-play reward statistics"
                        "[/bold cyan] "
                        f"[dim]update={update:,} iteration={collection_iteration} "
                        f"new_episodes={len(completed_rewards)} "
                        f"total_episodes={len(self_play_episode_rewards):,} "
                        f"window={len(recent_rewards)} "
                        f"reward_mean_10={recent_stats.mean:.2f} "
                        f"reward_median_10={recent_stats.median:.2f} "
                        f"reward_std_10={recent_stats.std:.2f} "
                        f"reward_min_10={min(recent_rewards):.2f} "
                        f"reward_max_10={max(recent_rewards):.2f} "
                        f"mode={collection_mode}[/dim]"
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
            recent_rewards = self_play_episode_rewards[-10:]
            recent_stats = EvaluationStats.from_rewards(tuple(recent_rewards))
            reward_summary = (
                f"full_episodes={len(self_play_episode_rewards):,} "
                f"reward_mean_10={recent_stats.mean:.2f} "
                f"reward_median_10={recent_stats.median:.2f} "
                f"reward_std_10={recent_stats.std:.2f} "
                f"reward_min_10={min(recent_rewards):.2f} "
                f"reward_max_10={max(recent_rewards):.2f}"
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
        wandb_exit_code = 0
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
        if batch_worker is not None:
            try:
                batch_worker.close()
            except Exception:
                pass
        if reanalysis_pipeline is not None:
            try:
                reanalysis_pipeline.close()
            except Exception:
                pass
        wandb_logger.finish(exit_code=wandb_exit_code)


if __name__ == "__main__":
    main()
