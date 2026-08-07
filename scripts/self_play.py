#!/usr/bin/env python3
"""Generate and print a short Atari self-play rollout."""

from __future__ import annotations

import argparse

import numpy as np
from rich import print
import torch
from torch import Tensor, nn

from atariagent import AtariAgent, GameTrajectory, SelfPlayWorker
from atariagent.search import MCTSConfig
from atariagent.selfplay import Environment, make_atari_environment


class UniformZeroPredictionNetwork(nn.Module):
    """Return equal policy logits and a symmetric zero-value distribution."""

    value_support_size = 601

    def __init__(self, action_space_size: int) -> None:
        super().__init__()
        if action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
        self.action_space_size = action_space_size

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = states.shape[0]
        policy_logits = states.new_zeros(batch_size, self.action_space_size)
        value_logits = states.new_zeros(batch_size, self.value_support_size)
        return policy_logits, value_logits


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def create_environments(
    env_id: str,
    *,
    num_envs: int,
    frame_stack: int,
    max_episode_steps: int,
) -> list[Environment]:
    """Create identically configured Atari environments."""
    environments: list[Environment] = []
    try:
        for _ in range(num_envs):
            environments.append(
                make_atari_environment(
                    env_id,
                    frame_stack=frame_stack,
                    max_episode_steps=max_episode_steps,
                )
            )
    except Exception:
        for environment in environments:
            environment.close()
        raise
    return environments


def format_floats(values: tuple[float, ...]) -> str:
    """Format a tuple of floats with exactly two decimal places."""
    return "(" + ", ".join(f"{value:.2f}" for value in values) + ")"


def print_trajectory(trajectory: GameTrajectory) -> None:
    """Print one trajectory without dumping pixel arrays or complete trees."""
    print(
        f"  episode={trajectory.episode_id} block={trajectory.block_id} "
        f"steps={len(trajectory)} terminated={trajectory.terminated} "
        f"truncated={trajectory.truncated}"
    )
    print(
        "    observation_shapes=",
        tuple(observation.shape for observation in trajectory.observations),
        sep="",
    )
    print(f"    actions={trajectory.actions}")
    print(f"    rewards={format_floats(trajectory.rewards)}")
    print(f"    raw_rewards={format_floats(trajectory.raw_rewards)}")
    print("    search_results=(")
    for result in trajectory.search_results:
        root_priors = tuple(
            child.prior for child in result.root.children.values()
        )
        print(
            "      "
            f"action={result.action}, root_priors={format_floats(root_priors)}, "
            f"visits={result.visit_counts}, "
            f"search_policy={format_floats(result.policy)}, "
            f"root_value={result.root_value:.2f}"
        )
    print("    )")


def run_self_play(
    *,
    env_id: str = "ALE/Pong-v5",
    num_envs: int = 4,
    steps: int = 100,
    frame_stack: int = 4,
    max_episode_steps: int = 3000,
    num_simulations: int = 50,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[tuple[GameTrajectory, ...], ...]:
    """Run the zero-initialized agent and return generated trajectories."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)
    np.random.seed(seed)
    environments = create_environments(
        env_id,
        num_envs=num_envs,
        frame_stack=frame_stack,
        max_episode_steps=max_episode_steps,
    )
    action_space_size = environments[0].action_space.n
    if any(
        environment.action_space.n != action_space_size
        for environment in environments
    ):
        for environment in environments:
            environment.close()
        raise ValueError("all environments must use the same action space")

    prediction_network = UniformZeroPredictionNetwork(action_space_size)
    agent = AtariAgent(
        frame_stack * 3,
        action_space_size,
        prediction_network=prediction_network,
        mcts_config=MCTSConfig(num_simulations=num_simulations),
    ).to(device)

    with SelfPlayWorker(
        agent,
        environments=environments,
        base_seed=seed,
        add_exploration_noise=False,
        temperature=1.0,
    ) as worker:
        return worker.run(steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a zero-value, uniform-policy Atari self-play rollout."
    )
    parser.add_argument("--game", default="ALE/Pong-v5")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories_by_environment = run_self_play(
        env_id=args.game,
        num_envs=args.num_envs,
        steps=args.steps,
        frame_stack=args.frame_stack,
        max_episode_steps=args.max_episode_steps,
        num_simulations=args.simulations,
        seed=args.seed,
        device=resolve_device(args.device),
    )

    for environment_index, trajectories in enumerate(
        trajectories_by_environment
    ):
        print(f"environment={environment_index} trajectories={len(trajectories)}")
        for trajectory in trajectories:
            print_trajectory(trajectory)


if __name__ == "__main__":
    main()
