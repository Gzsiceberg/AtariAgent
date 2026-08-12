#!/usr/bin/env python3
"""Benchmark production-shaped local Ray reanalysis actor pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import ray
import torch

from atariagent.models import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig
from atariagent.training import (
    ReanalysisPipeline,
    create_reanalysis_actors,
    initialize_local_ray,
    make_target_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-counts", default="1,2,4")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--action-space-size", type=int, default=18)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--actor-gpus", type=float, default=0.25)
    parser.add_argument("--actor-threads", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_batch(args: argparse.Namespace) -> ReplayBatch:
    batch_size = args.batch_size
    unroll_steps = args.unroll_steps
    states = unroll_steps + 1
    frames = torch.zeros(
        batch_size,
        4 + unroll_steps,
        3,
        96,
        96,
        dtype=torch.uint8,
    )
    return ReplayBatch(
        frames=frames,
        actions=torch.zeros(batch_size, unroll_steps, 1, dtype=torch.long),
        rewards=torch.zeros(batch_size, unroll_steps),
        policy_targets=torch.full(
            (batch_size, states, args.action_space_size),
            1.0 / args.action_space_size,
        ),
        value_targets=torch.zeros(batch_size, states),
        action_mask=torch.ones(batch_size, unroll_steps, dtype=torch.bool),
        target_mask=torch.ones(batch_size, states, dtype=torch.bool),
        value_mask=torch.ones(batch_size, states, dtype=torch.bool),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size),
        value_bootstrap_frames=frames.clone(),
        value_bootstrap_values=torch.zeros(batch_size, states),
        value_bootstrap_discounts=torch.ones(batch_size, states),
        value_bootstrap_mask=torch.ones(batch_size, states, dtype=torch.bool),
    )


def benchmark_count(
    count: int,
    args: argparse.Namespace,
    batch: ReplayBatch,
) -> dict[str, object]:
    representation = RepresentationNetwork(12)
    prediction = PredictionNetwork(args.action_space_size)
    dynamics = DynamicsNetwork(args.action_space_size)
    actors = create_reanalysis_actors(
        count=count,
        num_gpus=(args.actor_gpus if torch.cuda.is_available() else 0.0),
        num_cpus=args.actor_threads,
        mcts_threads=args.actor_threads,
        in_channels=12,
        action_space_size=args.action_space_size,
        mcts_config=MCTSConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=5,
        ),
        policy_enabled=True,
        rng_seed=0,
        support_min=-300,
        support_max=300,
        precision=args.precision,
    )
    pipeline = ReanalysisPipeline(
        actors,
        reanalyze_targets=True,
        policy_chunk_size=args.chunk_size,
        prefetch_batches=count,
        timeout_seconds=600.0,
        max_weight_lag=0,
        cache_targets=False,
    )
    try:
        pipeline.publish_weights(
            0,
            make_target_state(representation, prediction, dynamics),
            wait=True,
        )
        for _ in range(count):
            pipeline.submit(batch)
        for _ in range(count):
            pipeline.wait_next()

        submitted = 0
        completed = 0
        actor_times: list[float] = []
        request_latencies: list[float] = []
        started = perf_counter()
        while completed < args.requests:
            while submitted < args.requests and pipeline.needs_prefetch:
                pipeline.submit(batch)
                submitted += 1
            ready = pipeline.wait_next()
            actor_times.append(ready.actor_duration_ms)
            request_latencies.append(ready.queue_wait_ms)
            completed += 1
        elapsed = perf_counter() - started
        return {
            "actor_count": count,
            "elapsed_seconds": elapsed,
            "requests_per_second": args.requests / elapsed,
            "mean_actor_ms": mean(actor_times),
            "mean_request_latency_ms": mean(request_latencies),
            "peak_pending_batches": pipeline.max_observed_pending,
            "peak_pending_payload_bytes": pipeline.max_observed_pending_bytes,
        }
    finally:
        pipeline.close()


def main() -> None:
    args = parse_args()
    actor_counts = tuple(
        int(value.strip())
        for value in args.actor_counts.split(",")
        if value.strip()
    )
    if not actor_counts or any(count <= 0 for count in actor_counts):
        raise ValueError("actor counts must be positive")
    if args.requests <= 0:
        raise ValueError("requests must be positive")

    owns_ray = initialize_local_ray()
    try:
        batch = make_batch(args)
        measurements = [
            benchmark_count(count, args, batch) for count in actor_counts
        ]
        baseline = float(measurements[0]["requests_per_second"])
        for measurement in measurements:
            measurement["throughput_vs_first"] = (
                float(measurement["requests_per_second"]) / baseline
            )
        result = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "batch_size": args.batch_size,
            "unroll_steps": args.unroll_steps,
            "action_space_size": args.action_space_size,
            "num_simulations": args.num_simulations,
            "chunk_size": args.chunk_size,
            "requests": args.requests,
            "actor_gpus": args.actor_gpus,
            "actor_threads": args.actor_threads,
            "measurements": measurements,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if owns_ray and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
