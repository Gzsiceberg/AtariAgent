"""Hydra structured configuration for AtariAgent training."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

FINAL_EVALUATION_RAW_FRAMES = 108_000


@dataclass
class EnvironmentConfig:
    """Atari environment and preprocessing settings."""

    id: str = "ALE/Pong-v5"
    frame_stack: int = 4
    frame_skip: int = 4
    screen_size: int = 96
    max_episode_steps: int = 3_000
    grayscale: bool = False
    episodic_life: bool = True


@dataclass
class SelfPlayConfig:
    """Parallel self-play and tree-search settings."""

    num_envs: int = 4
    total_transitions: int = 100_000
    num_simulations: int = 16
    search_algorithm: str = "gumbel"
    num_top_actions: int = 4
    c_visit: float = 50.0
    c_scale: float = 0.1
    steps_per_iteration: int = 100
    trajectory_length: int = 400
    clip_rewards: bool = True
    add_exploration_noise: bool = True


@dataclass
class ReplayConfig:
    """FIFO replay settings."""

    max_transitions: int = 10_000
    warmup_transitions: int = 2_000
    priority_epsilon: float = 1e-6


@dataclass
class TrainingConfig:
    """Optimizer, unroll, and update settings."""

    device: str = "auto"
    steps: int = 100_000
    final_steps: int = 20_000
    updates_per_iteration: int = 400
    batch_size: int = 256
    unroll_steps: int = 5
    td_steps: int = 5
    lstm_horizon: int = 5
    value_target: str = "mixed"
    mixed_value_start_step: int = 30_000
    mixed_value_threshold: int = 5_000
    use_target_network_reanalysis: bool = True
    policy_reanalysis_chunk_size: int = 768
    cache_reanalyzed_targets: bool = True
    reanalysis_prefetch_batches: int = 2
    batch_max_in_flight: int = 3
    batch_ready_prefetch: int = 2
    batch_worker_timeout_seconds: float = 600.0
    reanalysis_timeout_seconds: float = 600.0
    reanalysis_worker_num_threads: int = 4
    target_update_interval_start: int = 200
    target_update_interval_end: int = 800
    target_update_interval_ramp_steps: int = 10_000
    target_update_interval_quantum: int = 100
    discount: float = 0.997
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_warmup_steps: int = 1_000
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 100_000
    max_gradient_norm: float = 5.0
    precision: str = "fp32"
    deterministic: bool = True
    runtime_type_checks: bool = True
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    log_every: int = 10


@dataclass
class AugmentationConfig:
    """EfficientZero image augmentation settings."""

    enabled: bool = True
    transforms: list[str] = field(default_factory=lambda: ["shift", "intensity"])
    shift_delta: int = 4
    intensity_scale: float = 0.05


@dataclass
class LossConfig:
    """Policy, value, value-prefix, and consistency loss settings."""

    policy_weight: float = 1.0
    value_weight: float = 0.25
    reward_weight: float = 1.0
    consistency_enabled: bool = True
    consistency_weight: float = 5.0


@dataclass
class CheckpointConfig:
    """Checkpoint destination and representative snapshot count."""

    path: str = "checkpoints/${environment_slug:${environment.id}}/agent_latest.pt"
    keep_representative: int = 10


@dataclass
class EvaluationConfig:
    """Noise-free checkpoint evaluation and output settings."""

    enabled: bool = True
    episodes: int = 10
    num_envs: int = 4
    data_path: str = (
        "evaluations/${environment_slug:${environment.id}}/agent_evaluations.json"
    )
    plot_path: str = (
        "evaluations/${environment_slug:${environment.id}}/agent_evaluation.png"
    )


@dataclass
class TrainAgentConfig:
    """Complete typed configuration for AtariAgent training."""

    seed: int = 0
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def environment_slug(environment_id: str) -> str:
    """Return the final, filesystem-safe name from an environment ID."""
    if not isinstance(environment_id, str):
        raise TypeError("environment_id must be a string")
    normalized = environment_id.strip()
    slug = normalized.rsplit("/", maxsplit=1)[-1]
    if slug in {"", ".", ".."}:
        raise ValueError("environment_id must contain a valid final component")
    return slug


def checkpoint_path_for_environment(environment_id: str) -> str:
    """Return the default latest-checkpoint path for an environment."""
    return f"checkpoints/{environment_slug(environment_id)}/agent_latest.pt"


def final_evaluation_max_episode_steps(frame_skip: int) -> int:
    """Return EfficientZero V1's 108k-frame final evaluation horizon."""
    if isinstance(frame_skip, bool) or not isinstance(frame_skip, int):
        raise TypeError("frame_skip must be an integer")
    if not 0 < frame_skip <= FINAL_EVALUATION_RAW_FRAMES:
        raise ValueError(
            "frame_skip must be positive and no greater than the raw-frame horizon"
        )
    return FINAL_EVALUATION_RAW_FRAMES // frame_skip


def target_network_update_interval(
    update: int,
    *,
    start: int,
    end: int,
    ramp_steps: int,
    quantum: int,
) -> int:
    """Return the quantized target-copy interval at ``update``."""
    for value, name in (
        (update, "update"),
        (start, "start"),
        (end, "end"),
        (ramp_steps, "ramp_steps"),
        (quantum, "quantum"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if update < 0:
        raise ValueError("update must be non-negative")
    if start <= 0 or end <= 0 or ramp_steps <= 0 or quantum <= 0:
        raise ValueError("target update schedule values must be positive")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if start % quantum != 0 or end % quantum != 0:
        raise ValueError("start and end must be divisible by quantum")
    if update >= ramp_steps:
        return end

    # Round the exact linear interpolation to the nearest quantum using
    # integer arithmetic, avoiding Python's ties-to-even round behavior.
    numerator = start * ramp_steps + (end - start) * update
    denominator = ramp_steps
    quantum_units = (
        2 * numerator + quantum * denominator
    ) // (2 * quantum * denominator)
    return min(max(quantum_units * quantum, start), end)


def target_network_update_steps(
    total_updates: int,
    *,
    start: int,
    end: int,
    ramp_steps: int,
    quantum: int,
) -> tuple[int, ...]:
    """Return all scheduled target-copy updates through ``total_updates``."""
    if isinstance(total_updates, bool) or not isinstance(total_updates, int):
        raise TypeError("total_updates must be an integer")
    if total_updates < 0:
        raise ValueError("total_updates must be non-negative")

    updates: list[int] = []
    previous_update = 0
    while True:
        interval = target_network_update_interval(
            previous_update,
            start=start,
            end=end,
            ramp_steps=ramp_steps,
            quantum=quantum,
        )
        next_update = previous_update + interval
        if next_update > total_updates:
            return tuple(updates)
        updates.append(next_update)
        previous_update = next_update


def next_collection_vector_steps(
    collected_transitions: int,
    total_transitions: int,
    num_envs: int,
    max_vector_steps: int,
) -> int:
    """Return vector steps that do not exceed the transition budget."""
    if num_envs <= 0 or max_vector_steps <= 0:
        raise ValueError("num_envs and max_vector_steps must be positive")
    if collected_transitions < 0 or total_transitions <= 0:
        raise ValueError("collected must be non-negative and total positive")
    if collected_transitions > total_transitions:
        raise ValueError("collected_transitions exceeds total_transitions")
    remaining = total_transitions - collected_transitions
    if remaining % num_envs != 0:
        raise ValueError("remaining transitions must be divisible by num_envs")
    return min(max_vector_steps, remaining // num_envs)


def visit_softmax_temperature(trained_steps: int, total_steps: int) -> float:
    """Return EfficientZero V1's schedule over collection-phase updates."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if trained_steps < 0:
        raise ValueError("trained_steps must be non-negative")
    if trained_steps < 0.5 * total_steps:
        return 1.0
    if trained_steps < 0.75 * total_steps:
        return 0.5
    return 0.25


def register_train_agent_config() -> None:
    """Register the structured schema and environment-path resolver."""
    OmegaConf.register_new_resolver(
        "environment_slug",
        environment_slug,
        replace=True,
    )
    ConfigStore.instance().store(
        name="train_agent_schema",
        node=TrainAgentConfig,
    )


__all__ = [
    "AugmentationConfig",
    "CheckpointConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "LossConfig",
    "ReplayConfig",
    "SelfPlayConfig",
    "TrainAgentConfig",
    "TrainingConfig",
    "checkpoint_path_for_environment",
    "environment_slug",
    "final_evaluation_max_episode_steps",
    "next_collection_vector_steps",
    "register_train_agent_config",
    "target_network_update_interval",
    "target_network_update_steps",
    "visit_softmax_temperature",
]
