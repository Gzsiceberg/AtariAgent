#!/usr/bin/env python3
"""Benchmark packed AtariAgent action selection with production MCTS."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch
from atariagent.search._tree_search_native import set_num_threads

from atariagent.agent import AtariAgent
from atariagent.search import SearchConfig
from atariagent.typecheck import set_runtime_typechecking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--action-space-size", type=int, default=18)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument(
        "--search-algorithm",
        choices=("puct", "gumbel"),
        default="puct",
    )
    parser.add_argument("--num-top-actions", type=int, default=4)
    parser.add_argument("--mcts-threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.action_space_size,
        args.num_simulations,
        args.mcts_threads,
        args.iterations,
    ) <= 0 or args.warmup < 0:
        raise ValueError("sizes and iterations must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    set_runtime_typechecking(False)
    set_num_threads(args.mcts_threads)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    agent = AtariAgent(
        12,
        args.action_space_size,
        search_config=SearchConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=5,
            search_algorithm=args.search_algorithm,
            num_top_actions=args.num_top_actions,
        ),
        search_rng=random.Random(0),
    ).to(device).eval()
    observations = torch.zeros(
        args.batch_size,
        12,
        96,
        96,
        device=device,
    )

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(args.warmup):
        agent.act(
            observations,
            add_exploration_noise=True,
            temperature=1.0,
        )
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    for _ in range(args.iterations):
        synchronize()
        started = perf_counter()
        agent.act(
            observations,
            add_exploration_noise=True,
            temperature=1.0,
        )
        synchronize()
        durations.append((perf_counter() - started) * 1_000.0)

    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "action_space_size": args.action_space_size,
        "search_algorithm": args.search_algorithm,
        "num_simulations": args.num_simulations,
        "num_top_actions": args.num_top_actions,
        "mcts_threads": args.mcts_threads,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "durations_ms": durations,
        "mean_ms": mean(durations),
        "median_ms": median(durations),
        "batches_per_second": 1_000.0 / mean(durations),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
