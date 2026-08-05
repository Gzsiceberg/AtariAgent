#!/usr/bin/env python3
"""Choose and play an Atari game with the keyboard.

Gymnasium's keyboard helper displays RGB frames in a Pygame window. Therefore,
the environment uses ``render_mode="rgb_array"`` internally rather than
``render_mode="human"``. The result is still an interactive human-play window;
ALE's literal ``human`` mode only displays frames and does not provide the
keyboard-to-action integration needed here.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

import ale_py
import gymnasium as gym
import pygame
from gymnasium.utils.play import play


POPULAR_GAMES = (
    "Pong",
    "Breakout",
    "SpaceInvaders",
    "MsPacman",
    "Asterix",
    "Seaquest",
)

# ALE describes joystick directions rather than movement on the screen. For
# example, Pong's RIGHT and LEFT actions move the paddle vertically.
DIRECTION_KEYS: dict[str, tuple[int, int]] = {
    "UP": (pygame.K_w, pygame.K_UP),
    "DOWN": (pygame.K_s, pygame.K_DOWN),
    "LEFT": (pygame.K_a, pygame.K_LEFT),
    "RIGHT": (pygame.K_d, pygame.K_RIGHT),
}


def available_games() -> dict[str, str]:
    """Return case-insensitive aliases mapped to canonical ALE v5 IDs."""
    gym.register_envs(ale_py)
    game_ids = sorted(
        spec.id
        for spec in gym.envs.registry.values()
        if spec.id.startswith("ALE/") and spec.id.endswith("-v5")
    )

    aliases: dict[str, str] = {}
    for game_id in game_ids:
        short_versioned_name = game_id.removeprefix("ALE/")
        short_name = short_versioned_name.removesuffix("-v5")
        aliases[game_id.casefold()] = game_id
        aliases[short_versioned_name.casefold()] = game_id
        aliases[short_name.casefold()] = game_id
    return aliases


def resolve_game(name: str, aliases: dict[str, str]) -> str:
    """Resolve Pong, Pong-v5, or ALE/Pong-v5 to a canonical environment ID."""
    try:
        return aliases[name.strip().casefold()]
    except KeyError as error:
        matching_names = sorted(
            {
                game_id.removeprefix("ALE/").removesuffix("-v5")
                for alias, game_id in aliases.items()
                if name.strip().casefold() in alias
            }
        )
        suggestion = (
            f" Possible matches: {', '.join(matching_names[:10])}."
            if matching_names
            else ""
        )
        raise ValueError(
            f"Unknown Atari game {name!r}.{suggestion} Use --list-games to list games."
        ) from error


def choose_game() -> str:
    """Prompt for a popular game number or any registered game name."""
    print("Choose an Atari game:")
    for index, game in enumerate(POPULAR_GAMES, start=1):
        print(f"  {index}. {game}")
    print("You can also type another game name. Press Enter for Pong.")

    choice = input("Game: ").strip()
    if not choice:
        return "Pong"
    if choice.isdigit() and 1 <= int(choice) <= len(POPULAR_GAMES):
        return POPULAR_GAMES[int(choice) - 1]
    return choice


def key_combinations(action_meaning: str) -> list[tuple[int, ...]]:
    """Convert an ALE action name into WASD/arrow/space combinations."""
    if action_meaning == "NOOP":
        return []

    uses_fire = action_meaning.endswith("FIRE")
    direction_name = (
        action_meaning.removesuffix("FIRE") if uses_fire else action_meaning
    )

    directions = [
        direction
        for direction in ("UP", "DOWN", "LEFT", "RIGHT")
        if direction in direction_name
    ]

    if not directions:
        return [(pygame.K_SPACE,)] if uses_fire else []

    wasd_keys = tuple(DIRECTION_KEYS[direction][0] for direction in directions)
    arrow_keys = tuple(DIRECTION_KEYS[direction][1] for direction in directions)
    if uses_fire:
        wasd_keys += (pygame.K_SPACE,)
        arrow_keys += (pygame.K_SPACE,)
    return [wasd_keys, arrow_keys]


def build_keyboard_mapping(action_meanings: list[str]) -> dict[tuple[int, ...], int]:
    """Build keyboard mappings for the actions exposed by the selected game."""
    mapping: dict[tuple[int, ...], int] = {}
    for action, meaning in enumerate(action_meanings):
        for keys in key_combinations(meaning):
            mapping[keys] = action
    return mapping


def format_keys(keys: tuple[int, ...]) -> str:
    return "+".join(
        "SPACE" if key == pygame.K_SPACE else pygame.key.name(key).upper()
        for key in keys
    )


def print_controls(
    action_meanings: list[str], mapping: dict[tuple[int, ...], int]
) -> None:
    controls_by_action: dict[int, list[str]] = {}
    for keys, action in mapping.items():
        controls_by_action.setdefault(action, []).append(format_keys(keys))

    print("\nControls for this game's action space:")
    print("  no key          -> NOOP")
    for action, meaning in enumerate(action_meanings):
        controls = controls_by_action.get(action)
        if controls:
            print(f"  {' or '.join(controls):<22} -> {meaning} (action {action})")
    print("  ESC / close window    -> quit")
    print("\nClick the game window once if it does not receive keyboard input.\n")


def episode_logger() -> Callable[..., None]:
    """Create a callback that prints the score at the end of every episode."""
    episode_reward = 0.0
    episode_steps = 0
    episode_number = 1

    def callback(
        _obs_t,
        _obs_tp1,
        _action,
        reward,
        terminated,
        truncated,
        _info,
    ) -> None:
        nonlocal episode_reward, episode_steps, episode_number
        episode_reward += float(reward)
        episode_steps += 1
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(
                f"Episode {episode_number}: score={episode_reward:g}, "
                f"steps={episode_steps}, {reason}"
            )
            episode_reward = 0.0
            episode_steps = 0
            episode_number += 1

    return callback


def list_games(aliases: dict[str, str]) -> None:
    names = sorted(
        {
            game_id.removeprefix("ALE/").removesuffix("-v5")
            for game_id in aliases.values()
        }
    )
    print("Available Atari games:")
    for name in names:
        print(f"  {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose and play a Gymnasium Atari game with the keyboard."
    )
    parser.add_argument(
        "--game",
        help="Game name or environment ID, for example Pong or ALE/Breakout-v5. "
        "If omitted, show an interactive game menu.",
    )
    parser.add_argument(
        "--list-games", action="store_true", help="List available games and exit."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Maximum agent steps per second (default: 30).",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=3.0,
        help="Display scale (default: 3).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reset seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aliases = available_games()

    if args.list_games:
        list_games(aliases)
        return 0
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.zoom <= 0:
        raise SystemExit("--zoom must be positive")

    requested_game = args.game or choose_game()
    try:
        game_id = resolve_game(requested_game, aliases)
    except ValueError as error:
        raise SystemExit(str(error)) from error

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

    # Gymnasium's play() owns the visible Pygame window. It requires rgb_array;
    # passing human here would raise an error because play() needs frame arrays.
    env = gym.make(game_id, render_mode="rgb_array")
    try:
        action_meanings = env.unwrapped.get_action_meanings()
        keyboard_mapping = build_keyboard_mapping(action_meanings)
        print(f"\nStarting {game_id}")
        print_controls(action_meanings, keyboard_mapping)

        play(
            env,
            keys_to_action=keyboard_mapping,
            noop=action_meanings.index("NOOP"),
            fps=args.fps,
            zoom=args.zoom,
            seed=args.seed,
            callback=episode_logger(),
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
