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
    discount: float = 0.997
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_warmup_steps: int = 1_000
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 100_000
    max_gradient_norm: float = 5.0
    log_every: int = 10


@dataclass
class LossConfig:
    """Policy, value, and value-prefix loss coefficients."""

    policy_weight: float = 1.0
    value_weight: float = 0.25
    reward_weight: float = 1.0


@dataclass
class CheckpointConfig:
    """Checkpoint destination and frequency."""

    path: str = "checkpoints/muzero_latest.pt"
    every: int = 1_000


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
    "LossConfig",
    "linear_priority_beta",
    "ReplayConfig",
    "SelfPlayConfig",
    "TrainingConfig",
    "TrainMuZeroConfig",
    "register_train_muzero_config",
    "visit_softmax_temperature",
]
