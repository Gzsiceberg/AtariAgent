#!/usr/bin/env python3
"""Evaluate a trained AtariAgent checkpoint on complete Atari episodes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from atariagent.evaluation import evaluate_agent, load_agent_checkpoint
from atariagent.selfplay import Environment, make_atari_environment
from atariagent.training.config import (
    EnvironmentConfig,
    checkpoint_path_for_environment,
    final_evaluation_max_episode_steps,
)


DEFAULT_ENVIRONMENT_ID = EnvironmentConfig().id


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print mean, median, and standard deviation of episode rewards."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=None,
        help="checkpoint path (default: derived from --environment)",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_option",
        type=Path,
        help="checkpoint path (alternative to the positional argument)",
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_ENVIRONMENT_ID,
        help=(
            "environment ID used to derive the default checkpoint path "
            f"(default: {DEFAULT_ENVIRONMENT_ID})"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="number of episodes (default: checkpoint evaluation setting or 10)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="parallel environments (default: checkpoint setting or 4)",
    )
    parser.add_argument("--seed", type=int, default=0, help="first episode seed")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, such as auto, cpu, or cuda (default: auto)",
    )
    return parser.parse_args()


def require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint has no valid {name} config")
    return value


def main() -> int:
    args = parse_args()
    if args.checkpoint is not None and args.checkpoint_option is not None:
        raise SystemExit("pass a checkpoint either positionally or with --checkpoint")
    checkpoint_path = args.checkpoint_option or args.checkpoint or Path(
        checkpoint_path_for_environment(args.environment)
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if args.episodes is not None and args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.num_envs is not None and args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = require_mapping(checkpoint.get("config"), "training")
    environment_config = require_mapping(config.get("environment"), "environment")
    evaluation_config = config.get("evaluation", {})
    if not isinstance(evaluation_config, Mapping):
        evaluation_config = {}
    episodes = args.episodes or int(evaluation_config.get("episodes", 10))
    num_envs = args.num_envs or int(evaluation_config.get("num_envs", 4))

    def environment_factory() -> Environment:
        frame_skip = int(environment_config["frame_skip"])
        return make_atari_environment(
            str(environment_config["id"]),
            frame_stack=int(environment_config["frame_stack"]),
            frame_skip=frame_skip,
            screen_size=int(environment_config["screen_size"]),
            max_episode_steps=final_evaluation_max_episode_steps(frame_skip),
            grayscale_obs=bool(environment_config["grayscale"]),
            terminal_on_life_loss=False,
        )

    probe_environment = environment_factory()
    try:
        action_space_size = int(probe_environment.action_space.n)
    finally:
        probe_environment.close()

    agent, _ = load_agent_checkpoint(
        checkpoint_path,
        action_space_size=action_space_size,
        device=device,
    )
    stats = evaluate_agent(
        agent,
        environment_factory,
        episodes=episodes,
        num_envs=num_envs,
        seed=args.seed,
        print_episode_results=True,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes:   {episodes}")
    print(f"Envs:       {min(num_envs, episodes)}")
    print(f"Mean:       {stats.mean:.3f}")
    print(f"Median:     {stats.median:.3f}")
    print(f"Std:        {stats.std:.3f}")
    print(f"Max:        {max(stats.rewards):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
