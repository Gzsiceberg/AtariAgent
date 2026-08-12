#!/usr/bin/env python3
"""Compare native and Python MCTS on production-shaped policy reanalysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from atariagent.models import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig
from atariagent.training.target import ValueTargetNetwork
from atariagent.typecheck import set_runtime_typechecking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--action-space-size", type=int, default=18)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
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
    frames = torch.zeros(
        batch_size,
        4 + unroll_steps,
        3,
        96,
        96,
        dtype=torch.uint8,
        device=device,
    )
    return ReplayBatch(
        frames=frames,
        actions=torch.zeros(
            batch_size,
            unroll_steps,
            1,
            dtype=torch.long,
            device=device,
        ),
        rewards=torch.zeros(batch_size, unroll_steps, device=device),
        policy_targets=torch.full(
            (batch_size, states, action_space_size),
            1.0 / action_space_size,
            device=device,
        ),
        value_targets=torch.zeros(batch_size, states, device=device),
        action_mask=torch.ones(
            batch_size,
            unroll_steps,
            dtype=torch.bool,
            device=device,
        ),
        policy_mask=torch.ones(
            batch_size,
            states,
            dtype=torch.bool,
            device=device,
        ),
        value_mask=torch.ones(
            batch_size,
            states,
            dtype=torch.bool,
            device=device,
        ),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size, device=device),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    target: ValueTargetNetwork,
    batch: ReplayBatch,
    *,
    search_method,
    chunk_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[list[float], int]:
    assert target._mcts is not None
    target._mcts.search_batch = search_method
    for _ in range(warmup):
        target.reanalyze_policies(batch, chunk_size=chunk_size)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations: list[float] = []
    for _ in range(iterations):
        synchronize(device)
        started = perf_counter()
        target.reanalyze_policies(batch, chunk_size=chunk_size)
        synchronize(device)
        durations.append(perf_counter() - started)
    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    return durations, peak_memory


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.unroll_steps <= 0:
        raise ValueError("batch size and unroll steps must be positive")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    set_runtime_typechecking(False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

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
        rng_seed=0,
        precision=args.precision,
    ).to(device).eval()
    assert target._mcts is not None
    native_search = target._mcts.search_batch
    python_search = target._mcts._search_batch_python

    native_times, native_peak = benchmark(
        target,
        batch,
        search_method=native_search,
        chunk_size=args.chunk_size,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    python_times, python_peak = benchmark(
        target,
        batch,
        search_method=python_search,
        chunk_size=args.chunk_size,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    native_mean = mean(native_times)
    python_mean = mean(python_times)
    result = {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "batch_size": args.batch_size,
        "unroll_steps": args.unroll_steps,
        "action_space_size": args.action_space_size,
        "num_simulations": args.num_simulations,
        "chunk_size": args.chunk_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "native_seconds": native_times,
        "python_seconds": python_times,
        "native_mean_seconds": native_mean,
        "python_mean_seconds": python_mean,
        "speedup": python_mean / native_mean,
        "time_reduction_percent": (1.0 - native_mean / python_mean) * 100.0,
        "native_peak_memory_bytes": native_peak,
        "python_peak_memory_bytes": python_peak,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
