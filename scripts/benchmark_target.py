#!/usr/bin/env python3
"""Benchmark packed target-network reanalysis directly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Callable

import torch

from atariagent.models import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig
from atariagent.search._mcts_native import set_num_threads
from atariagent.training.target import ValueTargetNetwork
from atariagent.typecheck import set_runtime_typechecking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--action-space-size", type=int, default=18)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=768)
    parser.add_argument("--mcts-threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_batch(
    *,
    batch_size: int,
    unroll_steps: int,
    action_space_size: int,
    device: torch.device,
) -> ReplayBatch:
    states = unroll_steps + 1
    stack_size = 4
    frames = torch.zeros(
        batch_size,
        stack_size + unroll_steps,
        3,
        96,
        96,
        dtype=torch.uint8,
        device=device,
    )
    return ReplayBatch(
        frames=frames,
        actions=torch.zeros(
            batch_size, unroll_steps, 1, dtype=torch.long, device=device
        ),
        rewards=torch.zeros(batch_size, unroll_steps, device=device),
        policy_targets=torch.full(
            (batch_size, states, action_space_size),
            1.0 / action_space_size,
            device=device,
        ),
        value_targets=torch.zeros(batch_size, states, device=device),
        action_mask=torch.ones(
            batch_size, unroll_steps, dtype=torch.bool, device=device
        ),
        policy_mask=torch.ones(
            batch_size, states, dtype=torch.bool, device=device
        ),
        value_mask=torch.ones(
            batch_size, states, dtype=torch.bool, device=device
        ),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size, device=device),
        value_bootstrap_frames=frames.clone(),
        value_bootstrap_values=torch.zeros(batch_size, states, device=device),
        value_bootstrap_discounts=torch.ones(batch_size, states, device=device),
        value_bootstrap_mask=torch.ones(
            batch_size, states, dtype=torch.bool, device=device
        ),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[list[float], int]:
    for _ in range(warmup):
        operation()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    for _ in range(iterations):
        synchronize(device)
        started = perf_counter()
        operation()
        synchronize(device)
        durations.append((perf_counter() - started) * 1_000.0)
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    return durations, peak


def summarize(durations: list[float], peak: int) -> dict[str, object]:
    return {
        "durations_ms": durations,
        "mean_ms": mean(durations),
        "median_ms": median(durations),
        "peak_memory_bytes": peak,
    }


def main() -> None:
    args = parse_args()
    positive = (
        args.batch_size,
        args.unroll_steps,
        args.action_space_size,
        args.num_simulations,
        args.chunk_size,
        args.mcts_threads,
        args.iterations,
    )
    if any(value <= 0 for value in positive) or args.warmup < 0:
        raise ValueError("sizes and iterations must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    set_runtime_typechecking(False)
    set_num_threads(args.mcts_threads)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    batch = make_batch(
        batch_size=args.batch_size,
        unroll_steps=args.unroll_steps,
        action_space_size=args.action_space_size,
        device=device,
    )
    target = ValueTargetNetwork(
        RepresentationNetwork(12),
        PredictionNetwork(args.action_space_size),
        dynamics=DynamicsNetwork(args.action_space_size),
        action_space_size=args.action_space_size,
        mcts_config=MCTSConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=5,
        ),
        precision=args.precision,
        chunk_size=args.chunk_size,
    ).to(device).eval()

    def optimized_policy() -> ReplayBatch:
        return target.reanalyze_policies(batch)

    def optimized_batch() -> ReplayBatch:
        return target.reanalyze_batch(batch)

    optimized_policy_times, optimized_policy_peak = benchmark(
        optimized_policy,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    optimized_batch_times, optimized_batch_peak = benchmark(
        optimized_batch,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "unroll_steps": args.unroll_steps,
        "policy_roots": args.batch_size * (args.unroll_steps + 1),
        "action_space_size": args.action_space_size,
        "num_simulations": args.num_simulations,
        "chunk_size": args.chunk_size,
        "mcts_threads": args.mcts_threads,
        "precision": args.precision,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "optimized_policy": summarize(
            optimized_policy_times, optimized_policy_peak
        ),
        "optimized_reanalyze_batch": summarize(
            optimized_batch_times, optimized_batch_peak
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
