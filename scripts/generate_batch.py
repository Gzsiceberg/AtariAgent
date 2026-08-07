#!/usr/bin/env python3
"""Generate self-play data, store it in FIFO replay, and print batch stats."""

from __future__ import annotations

from collections.abc import Iterable

import hydra
from omegaconf import DictConfig
from rich import print
from rich.table import Table
import numpy as np
import torch
from torch import Tensor, nn

from atariagent import AtariAgent, FIFOReplayBuffer, GameTrajectory, SelfPlayWorker
from atariagent.search import MCTSConfig
from atariagent.selfplay import Environment, make_atari_environment


class UniformZeroPredictionNetwork(nn.Module):
    """Return uniform policy logits and a zero-centered value distribution."""

    value_support_size = 601

    def __init__(self, action_space_size: int) -> None:
        super().__init__()
        self.action_space_size = action_space_size

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = states.shape[0]
        return (
            states.new_zeros(batch_size, self.action_space_size),
            states.new_zeros(batch_size, self.value_support_size),
        )


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def create_environments(config: DictConfig) -> list[Environment]:
    """Create the configured number of identically preprocessed games."""
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
                )
            )
    except Exception:
        for environment in environments:
            environment.close()
        raise
    return environments


def flatten_trajectories(
    trajectories_by_environment: tuple[tuple[GameTrajectory, ...], ...],
) -> Iterable[GameTrajectory]:
    """Yield trajectory blocks in environment order."""
    for trajectories in trajectories_by_environment:
        yield from trajectories


def tensor_summary(name: str, tensor: Tensor) -> tuple[str, ...]:
    """Return compact tensor metadata without materializing float pixel copies."""
    memory_mib = tensor.numel() * tensor.element_size() / (1024**2)
    if tensor.numel() == 0:
        value_range = "empty"
    elif tensor.dtype == torch.bool:
        value_range = f"true={int(tensor.sum())}/{tensor.numel()}"
    else:
        value_range = f"min={tensor.min().item():.3g} max={tensor.max().item():.3g}"
    return (
        name,
        str(tuple(tensor.shape)),
        str(tensor.dtype).removeprefix("torch."),
        f"{memory_mib:.2f}",
        value_range,
    )


def print_batch_stats(batch) -> None:
    """Print replay batch tensor shapes, dtypes, masks, and target statistics."""
    table = Table(title="Sampled replay batch")
    for heading in ("tensor", "shape", "dtype", "MiB", "values"):
        table.add_column(heading)

    tensor_names = (
        "observations",
        "actions",
        "rewards",
        "policy_targets",
        "root_values",
        "action_mask",
        "target_mask",
    )
    for name in tensor_names:
        table.add_row(*tensor_summary(name, getattr(batch, name)))
    print(table)

    valid_rewards = batch.rewards[batch.action_mask]
    valid_values = batch.root_values[batch.target_mask]
    valid_policies = batch.policy_targets[batch.target_mask]
    print(
        "batch summary: "
        f"size={batch.batch_size} unroll_steps={batch.unroll_steps} "
        f"valid_actions={int(batch.action_mask.sum())}/{batch.action_mask.numel()} "
        f"valid_targets={int(batch.target_mask.sum())}/{batch.target_mask.numel()} "
        f"reward_mean={valid_rewards.mean().item():.4f} "
        f"root_value_mean={valid_values.mean().item():.4f} "
        f"policy_row_sum_mean={valid_policies.sum(dim=-1).mean().item():.4f}"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="generate_batch")
def main(config: DictConfig) -> None:
    """Run outer self-play iterations and inner replay-sampling iterations."""
    if config.training.batch_size != 256:
        print(
            "[yellow]Note:[/yellow] EfficientZero V1 uses batch_size=256; "
            f"this run uses {config.training.batch_size}."
        )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.training.device)
    environments = create_environments(config)

    try:
        action_space_size = int(environments[0].action_space.n)
        if any(
            int(environment.action_space.n) != action_space_size
            for environment in environments
        ):
            raise ValueError("all environments must have the same action space")

        agent = AtariAgent(
            config.environment.frame_stack * 3,
            action_space_size,
            prediction_network=UniformZeroPredictionNetwork(action_space_size),
            mcts_config=MCTSConfig(
                num_simulations=config.self_play.num_simulations
            ),
        ).to(device)
        replay = FIFOReplayBuffer(
            config.replay.max_transitions, seed=config.seed
        )

        with SelfPlayWorker(
            agent,
            environments=environments,
            base_seed=config.seed,
            clip_rewards=True,
            add_exploration_noise=config.self_play.add_exploration_noise,
            temperature=config.self_play.temperature,
        ) as worker:
            for worker_iteration in range(
                1, config.self_play.worker_iterations + 1
            ):
                trajectories_by_environment = worker.run(
                    config.self_play.steps_per_iteration
                )
                trajectories = tuple(
                    flatten_trajectories(trajectories_by_environment)
                )
                insertion = replay.extend(trajectories)

                print(
                    f"\n[bold]worker_iteration={worker_iteration}[/bold] "
                    f"generated_trajectories={len(trajectories)} "
                    f"generated_transitions={sum(map(len, trajectories))} "
                    f"evicted_trajectories={insertion.evicted_trajectories} "
                    f"evicted_transitions={insertion.evicted_transitions}"
                )
                print(
                    f"replay: trajectories={replay.trajectory_count} "
                    f"transitions={len(replay)}/{replay.max_transitions} "
                    f"utilization={replay.utilization:.1%}"
                )

                for sample_iteration in range(
                    1, config.replay.sample_iterations + 1
                ):
                    batch = replay.sample(
                        config.training.batch_size,
                        unroll_steps=config.training.unroll_steps,
                    )
                    print(
                        f"[bold]worker_iteration={worker_iteration} "
                        f"sample_iteration={sample_iteration}[/bold]"
                    )
                    print_batch_stats(batch)
    except Exception:
        # SelfPlayWorker owns and closes the environments only after it exists.
        for environment in environments:
            environment.close()
        raise


if __name__ == "__main__":
    main()
