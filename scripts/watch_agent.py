#!/usr/bin/env python3
"""Watch a trained AtariAgent checkpoint play complete Atari episodes."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import torch

from atariagent.evaluation import evaluate_agent, load_agent_checkpoint
from atariagent.selfplay import Environment, make_atari_environment
from atariagent.training.config import (
    EnvironmentConfig,
    checkpoint_path_for_environment,
    final_evaluation_max_episode_steps,
)

DEFAULT_ENVIRONMENT_ID = EnvironmentConfig().id
ATARI_RAW_FRAMES_PER_SECOND = 60
WINDOW_SCALE = 3


class PygameDisplayEnvironment:
    """Display RGB-array renders without ALE's OpenGL ``human`` renderer."""

    def __init__(self, environment: Environment, *, frame_skip: int, title: str):
        self.environment = environment
        self.action_space = environment.action_space
        self.frame_rate = max(1, round(ATARI_RAW_FRAMES_PER_SECOND / frame_skip))
        self.title = title
        self.screen: pygame.Surface | None = None
        self.clock = pygame.time.Clock()

    def reset(self, *, seed: int | None = None):
        result = self.environment.reset(seed=seed)
        self._display_frame()
        return result

    def step(self, action: int):
        result = self.environment.step(action)
        self._display_frame()
        return result

    def _display_frame(self) -> None:
        frame = self.environment.render()  # type: ignore[attr-defined]
        if isinstance(frame, list):
            frame = frame[-1]
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise RuntimeError("Atari environment did not render an RGB frame")

        height, width = frame.shape[:2]
        window_size = (width * WINDOW_SCALE, height * WINDOW_SCALE)
        if self.screen is None:
            pygame.display.init()
            pygame.display.set_caption(self.title)
            self.screen = pygame.display.set_mode(window_size)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, window_size)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.frame_rate)

    def close(self) -> None:
        try:
            self.environment.close()
        finally:
            pygame.display.quit()


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` to an available PyTorch device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def require_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return a checkpoint configuration section or raise a useful error."""
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint has no valid {name} config")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a window and watch an AtariAgent checkpoint play Atari."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=None,
        help="checkpoint path (default: derived from --environment)",
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
        help="number of visible episodes (default: checkpoint setting or 1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="first episode seed")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, such as auto, cpu, or cuda (default: auto)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint or Path(
        checkpoint_path_for_environment(args.environment)
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if args.episodes is not None and args.episodes <= 0:
        raise SystemExit("--episodes must be positive")

    if (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        print(
            "Warning: DISPLAY and WAYLAND_DISPLAY are unset. The game window may "
            "not open in this terminal session.",
            file=sys.stderr,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = require_mapping(checkpoint.get("config"), "training")
    environment_config = require_mapping(config.get("environment"), "environment")
    evaluation_config = config.get("evaluation", {})
    if not isinstance(evaluation_config, Mapping):
        evaluation_config = {}
    episodes = args.episodes or int(evaluation_config.get("episodes", 1))

    def make_environment(*, visible: bool) -> Environment:
        frame_skip = int(environment_config["frame_skip"])
        environment = make_atari_environment(
            str(environment_config["id"]),
            frame_stack=int(environment_config["frame_stack"]),
            frame_skip=frame_skip,
            screen_size=int(environment_config["screen_size"]),
            max_episode_steps=final_evaluation_max_episode_steps(frame_skip),
            grayscale_obs=bool(environment_config["grayscale"]),
            terminal_on_life_loss=False,
            render_mode="rgb_array" if visible else None,
        )
        if not visible:
            return environment
        return PygameDisplayEnvironment(
            environment,
            frame_skip=frame_skip,
            title=f"AtariAgent — {environment_config['id']}",
        )

    probe_environment = make_environment(visible=False)
    try:
        action_space_size = int(probe_environment.action_space.n)
    finally:
        probe_environment.close()

    device = resolve_device(args.device)
    agent, _ = load_agent_checkpoint(
        checkpoint_path,
        action_space_size=action_space_size,
        device=device,
    )

    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Environment: {environment_config['id']}")
    print(f"Device:      {device}")
    print("Close the game window or press Ctrl-C to stop.\n")

    try:
        stats = evaluate_agent(
            agent,
            lambda: make_environment(visible=True),
            episodes=episodes,
            num_envs=1,
            seed=args.seed,
            print_episode_results=True,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130

    print(f"Episodes: {episodes}")
    print(f"Mean:     {stats.mean:.3f}")
    print(f"Median:   {stats.median:.3f}")
    print(f"Std:      {stats.std:.3f}")
    print(f"Max:      {max(stats.rewards):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
