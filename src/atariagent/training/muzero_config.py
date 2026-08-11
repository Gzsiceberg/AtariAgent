"""Hydra structured configuration for MuZero Atari training."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


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
    """Parallel self-play and MCTS settings."""

    num_envs: int = 4
    total_transitions: int = 100_000
    num_simulations: int = 50
    steps_per_iteration: int = 100
    trajectory_length: int = 400
    clip_rewards: bool = True
    add_exploration_noise: bool = True


@dataclass
class ReplayConfig:
    """FIFO replay settings."""

    max_transitions: int = 10_000
    warmup_transitions: int = 2_000
    priority_alpha: float = 0.6
    priority_beta_initial: float = 0.4
    priority_beta_final: float = 1.0
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
    use_target_network_reanalysis: bool = True
    policy_reanalysis_ratio: float = 0.99
    policy_reanalysis_chunk_size: int = 256
    reanalysis_start_step: int = 1_000
    target_update_interval: int = 200
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
    compile_mode: str = "default"
    pin_memory: bool = True
    log_every: int = 10


@dataclass
class LossConfig:
    """Policy, value, and value-prefix loss coefficients."""

    policy_weight: float = 1.0
    value_weight: float = 0.25
    reward_weight: float = 1.0


@dataclass
class CheckpointConfig:
    """Checkpoint destination, frequency, and representative snapshot count."""

    path: str = "checkpoints/muzero_latest.pt"
    every: int = 1_000
    keep_representative: int = 10


@dataclass
class EvaluationConfig:
    """Noise-free checkpoint evaluation and output settings."""

    enabled: bool = True
    episodes: int = 10
    data_path: str = "evaluations/muzero_evaluations.json"
    plot_path: str = "evaluations/muzero_evaluation.png"


@dataclass
class TrainMuZeroConfig:
    """Complete typed configuration for MuZero training."""

    seed: int = 0
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


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


def linear_priority_beta(
    trained_steps: int,
    training_steps: int,
    initial_beta: float,
    final_beta: float,
) -> float:
    """Linearly anneal the prioritized-replay importance exponent."""
    if training_steps <= 0:
        raise ValueError("training_steps must be positive")
    if trained_steps < 0:
        raise ValueError("trained_steps must be non-negative")
    if not 0.0 <= initial_beta <= final_beta <= 1.0:
        raise ValueError("priority betas must satisfy 0 <= initial <= final <= 1")
    fraction = min(trained_steps / training_steps, 1.0)
    return initial_beta + fraction * (final_beta - initial_beta)


def visit_softmax_temperature(trained_steps: int, training_steps: int) -> float:
    """Return EfficientZero's three-stage self-play temperature."""
    if training_steps <= 0:
        raise ValueError("training_steps must be positive")
    if trained_steps < 0:
        raise ValueError("trained_steps must be non-negative")
    if trained_steps < 0.5 * training_steps:
        return 1.0
    if trained_steps < 0.75 * training_steps:
        return 0.5
    return 0.25


def register_train_muzero_config() -> None:
    """Register the structured schema before Hydra composes the YAML file."""
    ConfigStore.instance().store(
        name="train_muzero_schema",
        node=TrainMuZeroConfig,
    )


__all__ = [
    "CheckpointConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "LossConfig",
    "linear_priority_beta",
    "next_collection_vector_steps",
    "ReplayConfig",
    "SelfPlayConfig",
    "TrainingConfig",
    "TrainMuZeroConfig",
    "register_train_muzero_config",
    "visit_softmax_temperature",
]
