#!/usr/bin/env python3
"""Train the recurrent dynamics model on random-action Atari rollouts."""

from dataclasses import dataclass

import ale_py
from einops import rearrange
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import hydra
from jaxtyping import Float
from omegaconf import DictConfig
import torch
from torch import Tensor

from atariagent.models import ConsistencyNetwork, DynamicsNetwork, RepresentationNetwork
from atariagent.training import DynamicsTrainer


@dataclass(frozen=True)
class RolloutBatch:
    observations: Float[Tensor, "batch sequence channels 96 96"]
    actions: Tensor
    rewards: Float[Tensor, "batch steps"]


def make_environment(env_id: str, frame_stack: int) -> gym.Env:
    """Create an Atari environment with standard preprocessing and stacking."""
    base_env = gym.make(
        env_id,
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=0.0,
    )
    processed_env = AtariPreprocessing(
        base_env,
        frame_skip=4,
        screen_size=96,
        terminal_on_life_loss=False,
        grayscale_obs=False,
        scale_obs=False,
    )
    return FrameStackObservation(processed_env, stack_size=frame_stack)


def tensor_observation(observation) -> Float[Tensor, "channels 96 96"]:
    """Convert stacked uint8 RGB frames to a channel-first float tensor."""
    frames = torch.as_tensor(observation).float() / 255.0
    return rearrange(
        frames,
        "stack height width channels -> (stack channels) height width",
    )


def collect_sequence(
    env: gym.Env,
    *,
    unroll_steps: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Collect one uninterrupted random-action sequence."""
    while True:
        observation, _ = env.reset(seed=seed)
        observations = [tensor_observation(observation)]
        actions: list[int] = []
        rewards: list[float] = []

        for _ in range(unroll_steps):
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

            observations.append(tensor_observation(next_observation))
            actions.append(action)
            rewards.append(float(reward))

        if len(actions) == unroll_steps:
            return (
                torch.stack(observations),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
            )
        seed += 1


def collect_batch(
    env: gym.Env,
    *,
    batch_size: int,
    unroll_steps: int,
    seed: int,
) -> RolloutBatch:
    sequences = [
        collect_sequence(
            env,
            unroll_steps=unroll_steps,
            seed=seed + batch_index,
        )
        for batch_index in range(batch_size)
    ]
    observations, actions, rewards = zip(*sequences, strict=True)
    return RolloutBatch(
        observations=torch.stack(observations),
        actions=rearrange(torch.stack(actions), "batch steps -> batch steps 1"),
        rewards=torch.stack(rewards),
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@hydra.main(version_base=None, config_path="../configs", config_name="train_dynamic")
def main(config: DictConfig) -> None:
    torch.manual_seed(config.seed)
    gym.register_envs(ale_py)
    env = make_environment(config.environment.id, config.environment.frame_stack)
    env.action_space.seed(config.seed)
    device = resolve_device(config.training.device)

    if config.training.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for batch normalization")

    representation = RepresentationNetwork(config.environment.frame_stack * 3).to(device)
    dynamics = DynamicsNetwork(env.action_space.n).to(device)
    consistency = ConsistencyNetwork(
        projection_dim=config.model.projection_dim,
        projection_hidden_dim=config.model.projection_hidden_dim,
        prediction_hidden_dim=config.model.prediction_hidden_dim,
    ).to(device)
    trainer = DynamicsTrainer(
        representation,
        dynamics,
        consistency,
        learning_rate=config.training.learning_rate,
        unroll_steps=config.training.unroll_steps,
        lstm_horizon=config.training.lstm_horizon,
        consistency_weight=config.training.consistency_weight,
    )

    try:
        for step in range(1, config.training.steps + 1):
            batch = collect_batch(
                env,
                batch_size=config.training.batch_size,
                unroll_steps=config.training.unroll_steps,
                seed=config.seed + step * config.training.batch_size,
            )
            metrics = trainer.train_step(
                batch.observations.to(device),
                batch.actions.to(device),
                batch.rewards.to(device),
            )
            if step == 1 or step % config.training.log_every == 0:
                print(
                    f"step={step:04d} loss={metrics.loss:.4f} "
                    f"reward={metrics.reward_loss:.4f} "
                    f"consistency={metrics.consistency_loss:.4f}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
