#!/usr/bin/env python3
"""Benchmark prepared replay insertion and batch construction."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import statistics
from time import perf_counter
from typing import Callable

import numpy as np

from atariagent.replay import FIFOReplayBuffer
from atariagent.search import SearchResult
from atariagent.selfplay import GameTrajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--trajectory-length", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--screen-size", type=int, default=96)
    parser.add_argument("--action-space-size", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def timed(call: Callable[[], object]) -> float:
    started = perf_counter()
    call()
    return (perf_counter() - started) * 1_000.0


def distribution(
    call: Callable[[], object], *, warmup: int, repetitions: int
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    values: list[float] = []
    gc.disable()
    try:
        for _ in range(repetitions):
            values.append(timed(call))
    finally:
        gc.enable()
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95)),
    }


def make_trajectory(
    *,
    sampleable_steps: int,
    lookahead_steps: int,
    block_id: int,
    stack_size: int,
    channels: int,
    screen_size: int,
    action_space_size: int,
) -> GameTrajectory:
    stored_steps = sampleable_steps + lookahead_steps
    frames = tuple(
        np.full(
            (channels, screen_size, screen_size),
            (block_id + offset) % 256,
            dtype=np.uint8,
        )
        for offset in range(stored_steps + stack_size)
    )
    result = SearchResult(
        action=0,
        visit_counts=tuple(range(1, action_space_size + 1)),
        root_value=0.25,
    )
    return GameTrajectory(
        environment_index=0,
        episode_id=block_id,
        block_id=block_id,
        stack_size=stack_size,
        frames=frames,
        actions=tuple(offset % action_space_size for offset in range(stored_steps)),
        rewards=tuple(float(offset % 3 - 1) for offset in range(stored_steps)),
        raw_rewards=tuple(0.0 for _ in range(stored_steps)),
        search_results=tuple(result for _ in range(stored_steps)),
        predicted_values=tuple(0.25 for _ in range(stored_steps)),
        terminated=False,
        truncated=False,
        full_episode_done=False,
        lookahead_steps=lookahead_steps,
    )


def replay_storage_bytes(replay: FIFOReplayBuffer) -> int:
    array_names = (
        "frames",
        "actions",
        "rewards",
        "policy_targets",
        "root_values",
        "predicted_values",
        "value_targets",
        "value_valid_mask",
    )
    return sum(
        getattr(trajectory, name).nbytes
        for trajectory in replay._trajectories
        for name in array_names
    ) + replay._transition_ids.nbytes + replay._priorities.nbytes


def main() -> None:
    args = parse_args()
    positive_values = (
        args.replay_size,
        args.trajectory_length,
        args.batch_size,
        args.stack_size,
        args.channels,
        args.screen_size,
        args.action_space_size,
        args.repetitions,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("sizes and repetitions must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.batch_size > args.replay_size:
        raise ValueError("batch size must not exceed replay size")

    lookahead_steps = max(args.unroll_steps, args.td_steps)
    replay = FIFOReplayBuffer(
        args.replay_size,
        unroll_steps=args.unroll_steps,
        td_steps=args.td_steps,
        discount=args.discount,
        seed=args.seed,
    )

    insertion_started = perf_counter()
    remaining = args.replay_size
    block_id = 0
    while remaining:
        sampleable_steps = min(args.trajectory_length, remaining)
        replay.add(
            make_trajectory(
                sampleable_steps=sampleable_steps,
                lookahead_steps=lookahead_steps,
                block_id=block_id,
                stack_size=args.stack_size,
                channels=args.channels,
                screen_size=args.screen_size,
                action_space_size=args.action_space_size,
            )
        )
        remaining -= sampleable_steps
        block_id += 1
    insertion_ms = (perf_counter() - insertion_started) * 1_000.0

    cold_sample_ms = timed(lambda: replay.sample(args.batch_size))
    warm_sample = distribution(
        lambda: replay.sample(args.batch_size),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    warm_sample_without_bootstraps = distribution(
        lambda: replay.sample(
            args.batch_size, include_value_bootstraps=False
        ),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    locations, transition_ids, importance_weights = replay._sample_context(
        args.batch_size
    )
    full_arrays = replay._allocate_batch_arrays(
        args.batch_size, include_value_bootstraps=True
    )
    arrays_without_bootstraps = replay._allocate_batch_arrays(
        args.batch_size, include_value_bootstraps=False
    )

    def fill(arrays: dict[str, np.ndarray]) -> None:
        replay._fill_batch_arrays(
            arrays,
            locations,
            transition_ids=transition_ids,
            importance_weights=importance_weights,
        )

    fill_full = distribution(
        lambda: fill(full_arrays),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    fill_without_bootstraps = distribution(
        lambda: fill(arrays_without_bootstraps),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    sample_context = distribution(
        lambda: replay._sample_context(args.batch_size),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    result = {
        "replay_size": args.replay_size,
        "trajectory_length": args.trajectory_length,
        "trajectory_count": replay.trajectory_count,
        "lookahead_steps": lookahead_steps,
        "batch_size": args.batch_size,
        "unroll_steps": args.unroll_steps,
        "td_steps": args.td_steps,
        "discount": args.discount,
        "stack_size": args.stack_size,
        "frame_shape": [args.channels, args.screen_size, args.screen_size],
        "action_space_size": args.action_space_size,
        "insertion_total_ms": insertion_ms,
        "insertion_per_trajectory_ms": insertion_ms / replay.trajectory_count,
        "cold_sample_ms": cold_sample_ms,
        "warm_sample": warm_sample,
        "warm_sample_without_bootstraps": warm_sample_without_bootstraps,
        "fill": fill_full,
        "fill_without_bootstraps": fill_without_bootstraps,
        "sample_context": sample_context,
        "replay_storage_bytes": replay_storage_bytes(replay),
        "maximum_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
