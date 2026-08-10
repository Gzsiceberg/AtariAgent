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


@dataclass
class SelfPlayConfig:
    """Parallel self-play and MCTS settings."""

    num_envs: int = 4
    num_simulations: int = 50
    steps_per_iteration: int = 100
    trajectory_length: int = 100
    clip_rewards: bool = True
    add_exploration_noise: bool = True
    temperature: float = 1.0


@dataclass
class ReplayConfig:
    """FIFO replay settings."""

    max_transitions: int = 100_000
    warmup_transitions: int = 1_000


@dataclass
class TrainingConfig:
    """Optimizer, unroll, and update settings."""

    device: str = "auto"
    steps: int = 10_000
    updates_per_iteration: int = 100
    batch_size: int = 256
    unroll_steps: int = 5
    td_steps: int = 5
    lstm_horizon: int = 5
    discount: float = 0.997
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_warmup_steps: int = 100
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 100_000
    recurrent_gradient_scale: float = 0.5
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
    "ReplayConfig",
    "SelfPlayConfig",
    "TrainingConfig",
    "TrainMuZeroConfig",
    "register_train_muzero_config",
]
