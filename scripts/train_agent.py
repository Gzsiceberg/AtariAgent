#!/usr/bin/env python3
"""Train AtariAgent from FIFO self-play replay."""

from __future__ import annotations

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
    proportional_training_update,
    register_train_agent_config,
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


def configure_training_backend(
    deterministic: bool, *, cudnn_benchmark: bool = False
) -> None:
    """Configure process-wide CUDA backends before starting native workers."""
    if not torch.cuda.is_available():
        return
    # Variable cache-miss root counts otherwise repeatedly trigger cuDNN
    # algorithm benchmarking and device-wide synchronization. Keep this a
    # startup setting: changing it around native requests also affects learning.
    torch.backends.cudnn.benchmark = cudnn_benchmark
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
                    time_limit_mode=config.environment.time_limit_mode,
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
        "consistency": trainer.consistency_network.state_dict(),
        "target_network": dict(target_state),
        "target_version": target_version,
        "optimizer": trainer.optimizer.state_dict(),
        "trainer_step": trainer.step_count,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    torch.save(checkpoint, path)


def capture_rng_state() -> dict[str, object]:
    """Capture process RNGs used by learner augmentation and sampling."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore process RNGs from a trusted pre-final snapshot."""
    try:
        random.setstate(state["python"])  # type: ignore[arg-type]
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
        torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pre-final snapshot has invalid RNG state") from error


def save_pre_final_snapshot(
    path: Path,
    *,
    agent: AtariAgent,
    trainer: Trainer,
    replay: FIFOReplayBuffer,
    target_state: Mapping[str, torch.Tensor],
    target_version: int,
    update: int,
    config: TrainAgentConfig,
    rng_state: Mapping[str, object],
) -> None:
    """Atomically persist everything required by the learner-only phase."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    snapshot: dict[str, object] = {
        "snapshot_type": "atariagent_pre_final",
        "snapshot_version": 1,
        "update": update,
        "representation": agent.representation_network.state_dict(),
        "dynamics": agent.dynamics_network.state_dict(),
        "prediction": agent.prediction_network.state_dict(),
        "consistency": trainer.consistency_network.state_dict(),
        "trainer": trainer.training_state_dict(),
        "target_network": dict(target_state),
        "target_version": target_version,
        "replay": replay.state_dict(tensor_arrays=True),
        "search_rng": agent.search.rng.getstate(),
        "rng": dict(rng_state),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    try:
        torch.save(snapshot, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_pre_final_snapshot(
    path: Path,
    *,
    agent: AtariAgent,
    trainer: Trainer,
    replay: FIFOReplayBuffer,
    expected_update: int,
) -> tuple[int, dict[str, torch.Tensor], int, Mapping[str, object]]:
    """Load a trusted pre-final snapshot into initialized training objects."""
    if not path.is_file():
        raise FileNotFoundError(f"pre-final snapshot not found: {path}")
    snapshot = torch.load(
        path,
        map_location="cpu",
        # Legacy replay arrays and RNG state require Python pickle. Only load
        # trusted snapshots produced locally by this training script.
        weights_only=False,
    )
    if not isinstance(snapshot, Mapping):
        raise TypeError("pre-final snapshot must contain a mapping")
    if snapshot.get("snapshot_type") != "atariagent_pre_final":
        raise ValueError("file is not an AtariAgent pre-final snapshot")
    if snapshot.get("snapshot_version") != 1:
        raise ValueError("unsupported pre-final snapshot version")

    update = snapshot.get("update")
    target_version = snapshot.get("target_version")
    if isinstance(update, bool) or not isinstance(update, int):
        raise TypeError("snapshot update must be an integer")
    if update != expected_update:
        raise ValueError(
            f"snapshot update {update} does not match training.steps {expected_update}"
        )
    if isinstance(target_version, bool) or not isinstance(target_version, int):
        raise TypeError("snapshot target_version must be an integer")

    for name, network in (
        ("representation", agent.representation_network),
        ("dynamics", agent.dynamics_network),
        ("prediction", agent.prediction_network),
    ):
        state = snapshot.get(name)
        if not isinstance(state, Mapping):
            raise TypeError(f"snapshot {name} state must be a mapping")
        network.load_state_dict(state)
    consistency_state = snapshot.get("consistency")
    if not isinstance(consistency_state, Mapping):
        raise TypeError("snapshot consistency state must be a mapping")
    trainer.consistency_network.load_state_dict(consistency_state)

    trainer_state = snapshot.get("trainer")
    replay_state = snapshot.get("replay")
    target_state = snapshot.get("target_network")
    search_rng_state = snapshot.get("search_rng")
    rng_state = snapshot.get("rng")
    if not isinstance(trainer_state, Mapping):
        raise TypeError("snapshot trainer state must be a mapping")
    if not isinstance(replay_state, Mapping):
        raise TypeError("snapshot replay state must be a mapping")
    if not isinstance(target_state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in target_state.items()
    ):
        raise TypeError("snapshot target-network state is invalid")
    if not isinstance(rng_state, Mapping):
        raise TypeError("snapshot RNG state must be a mapping")
    trainer.load_training_state_dict(trainer_state)
    if trainer.step_count != update:
        raise ValueError("snapshot trainer step does not match its update")
    replay.load_state_dict(replay_state)
    try:
        agent.search.rng.setstate(search_rng_state)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("snapshot search RNG state is invalid") from error
    return update, dict(target_state), target_version, rng_state


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
    if config.training.visit_softmax_temperature_horizon not in {
        "collect_steps", "total_steps"
    }:
        raise ValueError(
            "training.visit_softmax_temperature_horizon must be "
            "collect_steps or total_steps"
        )
    if config.replay.per_mode not in {"v1", "v2"}:
        raise ValueError("replay.per_mode must be v1 or v2")
    if not 0.0 <= config.replay.priority_alpha <= 1.0:
        raise ValueError("replay.priority_alpha must be in [0, 1]")
    if not (
        0.0
        <= config.replay.priority_beta_initial
        <= config.replay.priority_beta_final
        <= 1.0
    ):
        raise ValueError(
            "replay priority beta bounds must satisfy 0 <= initial <= final <= 1"
        )
    if config.training.steps <= 0:
        raise ValueError("training.steps must be positive")
    if config.training.final_steps < 0:
        raise ValueError("training.final_steps must be non-negative")
    if config.training.updates_per_iteration <= 0:
        raise ValueError("updates_per_iteration must be positive")
    if config.training.log_every <= 0:
        raise ValueError("log_every must be positive")
    if config.training.progress_mode not in {"auto", "always", "never"}:
        raise ValueError("training.progress_mode must be auto, always, or never")
    if config.training.progress_interval_seconds <= 0.0:
        raise ValueError("training.progress_interval_seconds must be positive")
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
    for value, name in (
        (config.reanalysis.target_update_interval, "target_update_interval"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"reanalysis.{name} must be an integer")
        if value <= 0:
            raise ValueError(f"reanalysis.{name} must be positive")
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
    if config.loss.consistency_weight < 0.0:
        raise ValueError("loss.consistency_weight must be non-negative")
    if config.checkpoint.collection_interval <= 0:
        raise ValueError("checkpoint.collection_interval must be positive")
    if config.checkpoint.final_interval <= 0:
        raise ValueError("checkpoint.final_interval must be positive")
    for value, name in (
        (
            config.checkpoint.pre_final_snapshot_path,
            "pre_final_snapshot_path",
        ),
        (config.checkpoint.resume_pre_final_path, "resume_pre_final_path"),
    ):
        if value is not None and not value.strip():
            raise ValueError(f"checkpoint.{name} must be null or non-empty")
    if not isinstance(config.evaluation.evaluate_on_resume, bool):
        raise TypeError("evaluation.evaluate_on_resume must be a boolean")
    if config.evaluation.evaluate_on_resume and not config.evaluation.enabled:
        raise ValueError(
            "evaluation.evaluate_on_resume requires evaluation.enabled=true"
        )
    if (
        config.evaluation.evaluate_on_resume
        and config.checkpoint.resume_pre_final_path is None
    ):
        raise ValueError(
            "evaluation.evaluate_on_resume requires "
            "checkpoint.resume_pre_final_path"
        )
    if config.evaluation.enabled and config.evaluation.episodes <= 0:
        raise ValueError("evaluation.episodes must be positive")
    if config.evaluation.enabled and config.evaluation.num_envs <= 0:
        raise ValueError("evaluation.num_envs must be positive")
    if not isinstance(config.wandb.enabled, bool):
        raise TypeError("wandb.enabled must be a boolean")
    if config.wandb.enabled and not config.wandb.project.strip():
        raise ValueError("wandb.project must not be empty when enabled")
    if config.wandb.name is not None and not config.wandb.name.strip():
        raise ValueError("wandb.name must not be empty")
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
    configure_training_backend(
        config.training.deterministic,
        cudnn_benchmark=config.training.cudnn_benchmark,
    )
    set_runtime_typechecking(config.training.runtime_type_checks)
    device = resolve_device(config.training.device)
    resume_path = (
        None
        if config.checkpoint.resume_pre_final_path is None
        else Path(config.checkpoint.resume_pre_final_path)
    )
    resuming_final_phase = resume_path is not None
    environments: list[Environment] = []
    reanalysis_pipeline: ReanalysisPipeline | None = None
    batch_worker: BatchWorker | None = None
    wandb_logger = WandbLogger()
    wandb_exit_code = 1

    try:
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
            run_name=(config.wandb.name or wandb_run_name(config.environment.id)),
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
                root_exploration_fraction=(
                    config.self_play.root_exploration_fraction
                ),
                num_top_actions=num_top_actions,
                c_visit=config.self_play.c_visit,
                c_scale=config.self_play.c_scale,
            ),
            search_rng=random.Random(config.seed),
            precision=config.training.precision,
            action_embedding=config.model.action_embedding,
        ).to(device)
        trainer = Trainer(
            agent.representation_network,
            agent.dynamics_network,
            agent.prediction_network,
            consistency_network=ConsistencyNetwork().to(device),
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
            per_mode=config.replay.per_mode,
            priority_alpha=config.replay.priority_alpha,
            priority_beta=config.replay.priority_beta_initial,
            priority_epsilon=config.replay.priority_epsilon,
            seed=config.seed,
        )
        target_state = make_target_state(
            agent.representation_network,
            agent.prediction_network,
            agent.dynamics_network,
        )
        target_version = 0
        update = 0
        resume_rng_state: Mapping[str, object] | None = None
        if resume_path is not None:
            (
                update,
                target_state,
                target_version,
                resume_rng_state,
            ) = load_pre_final_snapshot(
                resume_path,
                agent=agent,
                trainer=trainer,
                replay=replay,
                expected_update=config.training.steps,
            )
            log(
                "[bold green]Pre-final snapshot loaded[/bold green] "
                f"[dim]update={update:,} replay={len(replay):,} "
                f"path={resume_path}[/dim]"
            )

        def create_reanalysis_pipeline() -> ReanalysisPipeline:
            pipeline = ReanalysisPipeline(
                in_channels=config.environment.frame_stack * image_channels,
                action_space_size=action_space_size,
                search_config=agent.search.config,
                policy_chunk_size=config.reanalysis.policy_chunk_size,
                action_embedding=config.model.action_embedding,
                cache_targets=config.reanalysis.cache_targets,
                cache_target_ttl=config.reanalysis.cache_target_ttl,
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
                device=device,
            )
            pipeline.publish_weights(
                target_version,
                target_state,
                wait=True,
            )
            return pipeline

        reanalysis_pipeline = create_reanalysis_pipeline()

        total_updates = config.training.steps + config.training.final_steps
        representative_updates = set(
            representative_checkpoint_updates(
                config.training.steps,
                config.training.final_steps,
                collection_interval=config.checkpoint.collection_interval,
                final_interval=config.checkpoint.final_interval,
            )
        )
        evaluation_records: list[EvaluationRecord] = []
        checkpointed_updates: set[int] = set()
        reward_tracker = EpisodeRewardTracker()
        self_play_episode_rewards: list[float] = []
        latest_checkpoint_path = Path(config.checkpoint.path)
        if config.evaluation.enabled:
            write_evaluation_history(
                config.evaluation.data_path,
                evaluation_records,
                environment_id=config.environment.id,
            )
        progress_disabled = config.training.progress_mode == "never" or (
            config.training.progress_mode == "auto" and not sys.stdout.isatty()
        )
        progress_options = {
            "dynamic_ncols": True,
            "file": sys.stdout,
            "disable": progress_disabled,
            "mininterval": config.training.progress_interval_seconds,
        }
        training_progress = tqdm(
            total=total_updates,
            initial=update,
            desc="Training",
            unit="update",
            position=0,
            **progress_options,
        )
        self_play_progress = tqdm(
            total=config.self_play.total_transitions,
            initial=(config.self_play.total_transitions if resuming_final_phase else 0),
            desc="Self-play",
            unit="transition",
            position=1,
            **progress_options,
        )
        log(
            "[bold cyan]AtariAgent training started[/bold cyan] "
            f"[dim]env={config.environment.id} device={device} "
            f"precision={config.training.precision} "
            f"per={config.replay.per_mode} "
            f"deterministic={config.training.deterministic} "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark} "
            f"compile={config.training.compile_model} "
            f"transitions={config.self_play.total_transitions:,} "
            f"updates={total_updates:,}[/dim]"
        )

        def evaluate_current_agent(
            checkpoint_path: Path, *, phase: str = "Evaluation"
        ) -> None:
            """Evaluate the in-memory agent and persist its result."""
            stats = evaluate_agent(
                agent,
                lambda: create_evaluation_environment(config),
                episodes=config.evaluation.episodes,
                num_envs=config.evaluation.num_envs,
                seed=config.seed,
            )
            evaluation_records.append(
                EvaluationRecord.create(update, checkpoint_path, stats)
            )
            write_evaluation_history(
                config.evaluation.data_path,
                evaluation_records,
                environment_id=config.environment.id,
            )
            wandb_logger.log_evaluation(stats, update=update)
            log(
                f"[bold blue]{phase} complete[/bold blue] "
                f"[dim]update={update:,} episodes={config.evaluation.episodes} "
                f"mean={stats.mean:.2f} median={stats.median:.2f} "
                f"std={stats.std:.2f} max={max(stats.rewards):.2f}[/dim]"
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
            evaluate_current_agent(saved_path)

        if resuming_final_phase and config.evaluation.evaluate_on_resume:
            assert resume_path is not None
            evaluate_current_agent(resume_path, phase="Resume evaluation")

        def create_batch_worker() -> BatchWorker:
            return BatchWorker(
                replay,
                batch_size=config.training.batch_size,
                device=device,
                reanalysis_pipeline=reanalysis_pipeline,
                value_target=config.training.value_target,
                collection_steps=config.training.steps,
                mixed_value_start_step=(config.training.mixed_value_start_step),
                mixed_value_threshold=config.training.mixed_value_threshold,
                priority_beta_initial=config.replay.priority_beta_initial,
                priority_beta_final=config.replay.priority_beta_final,
                priority_beta_steps=(
                    config.training.steps + config.training.final_steps
                ),
                max_in_flight=config.training.batch_max_in_flight,
                ready_prefetch=config.training.batch_ready_prefetch,
                timeout_seconds=(config.training.batch_worker_timeout_seconds),
            )

        batch_worker = create_batch_worker()
        if resume_rng_state is not None:
            restore_rng_state(resume_rng_state)

        def apply_gpu_update(ready: ReadyBatch) -> None:
            nonlocal update, target_state, target_version
            metrics = trainer.train_step(ready.gpu_batch)
            # Priority transfer synchronizes the learner stream, so it also
            # makes the pinned H2D source safe to release immediately.
            batch_worker.complete(ready, metrics.priorities)

            update += 1
            if target_network_update_due(
                update,
                last_update=target_version,
                interval=config.reanalysis.target_update_interval,
            ):
                target_state = make_target_state(
                    agent.representation_network,
                    agent.prediction_network,
                    agent.dynamics_network,
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

        def run_updates_to(target_update: int) -> None:
            """Advance in bounded chunks before collecting more experience."""
            if target_update < update:
                raise ValueError("target update must not precede current update")
            while update < target_update:
                run_updates(
                    min(
                        config.training.updates_per_iteration,
                        target_update - update,
                    )
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
            lookahead_steps=max(
                config.training.unroll_steps,
                config.training.td_steps,
            ),
            base_seed=config.seed,
            clip_rewards=config.self_play.clip_rewards,
        ) as worker:
            search_mode = agent.search.config.search_algorithm.upper()
            while (
                not resuming_final_phase
                and worker.total_transitions < config.self_play.total_transitions
            ):
                collection_iteration += 1
                vector_steps = next_collection_vector_steps(
                    worker.total_transitions,
                    config.self_play.total_transitions,
                    config.self_play.num_envs,
                    config.self_play.steps_per_iteration,
                )
                warming_up = len(replay) < minimum_replay_size
                previous_transitions = worker.total_transitions
                if agent.search.config.search_algorithm == "puct":
                    temperature = visit_softmax_temperature(
                        update,
                        config.training.steps,
                        final_steps=config.training.final_steps,
                        horizon=config.training.visit_softmax_temperature_horizon,
                    )
                    root_noise_temperature = 1.0
                    grouped = worker.run(
                        vector_steps,
                        temperature=temperature,
                        root_noise_temperature=root_noise_temperature,
                        gumbel_sampling=False,
                    )
                else:
                    temperature = None
                    root_noise_temperature = None
                    grouped = worker.run(
                        vector_steps,
                        root_noise_temperature=0.0,
                        gumbel_sampling=True,
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
                    "mode": search_mode,
                }
                if temperature is not None:
                    progress_stats["temperature"] = f"{temperature:.2f}"
                if root_noise_temperature is not None:
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
                        f"mode={search_mode}[/dim]"
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

                run_updates_to(
                    proportional_training_update(
                        worker.total_transitions,
                        config.self_play.total_transitions,
                        config.training.steps,
                    )
                )

            # Deliberately discard in-progress partial trajectories at the
            # collection budget. EfficientZero actors leave these unfinished
            # trajectories out of replay rather than flushing them immediately
            # before the learner-only phase.

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
        # Normally already exact from proportional pacing; retain this as a
        # safety net when replay only becomes trainable after collection ends.
        run_updates_to(config.training.steps)

        pre_final_path = (
            None
            if config.checkpoint.pre_final_snapshot_path is None
            else Path(config.checkpoint.pre_final_snapshot_path)
        )
        if not resuming_final_phase and pre_final_path is not None:
            if update != config.training.steps:
                raise RuntimeError("pre-final snapshot boundary was not reached")
            pre_final_rng_state = capture_rng_state()
            save_pre_final_snapshot(
                pre_final_path,
                agent=agent,
                trainer=trainer,
                replay=replay,
                target_state=target_state,
                target_version=target_version,
                update=update,
                config=config,
                rng_state=pre_final_rng_state,
            )
            log(
                "[bold green]Pre-final snapshot saved[/bold green] "
                f"[dim]update={update:,} replay={len(replay):,} "
                f"path={pre_final_path}[/dim]"
            )

        if not resuming_final_phase and pre_final_path is not None:
            # Restart workers after the snapshot so no prefetched
            # collection-phase batch enters final training.
            boundary_rng_state = capture_rng_state()
            batch_worker.close()
            batch_worker = None
            reanalysis_pipeline.close()
            reanalysis_pipeline = create_reanalysis_pipeline()
            batch_worker = create_batch_worker()
            restore_rng_state(boundary_rng_state)

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
