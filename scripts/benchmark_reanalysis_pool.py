#!/usr/bin/env python3
"""Benchmark native reanalysis at several bounded queue depths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from atariagent.models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from atariagent.replay_batch import ReplayBatch
from atariagent.search import SearchConfig
from atariagent.training import ReanalysisPipeline, make_target_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefetch-depths", default="1,2,4")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--action-space-size", type=int, default=18)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=768)
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_batch(args: argparse.Namespace) -> ReplayBatch:
    states = args.unroll_steps + 1
    frame_count = 4 + args.unroll_steps
    reanalysis_frames = torch.zeros(
        args.batch_size,
        frame_count + args.unroll_steps,
        3,
        96,
        96,
        dtype=torch.uint8,
    )
    frames = reanalysis_frames[:, :frame_count]
    bootstrap_frames = reanalysis_frames[:, args.unroll_steps :]
    return ReplayBatch(
        frames=frames,
        actions=torch.zeros(args.batch_size, args.unroll_steps, 1, dtype=torch.long),
        rewards=torch.zeros(args.batch_size, args.unroll_steps),
        policy_targets=torch.full(
            (args.batch_size, states, args.action_space_size),
            1.0 / args.action_space_size,
        ),
        value_targets=torch.zeros(args.batch_size, states),
        action_mask=torch.ones(args.batch_size, args.unroll_steps, dtype=torch.bool),
        policy_mask=torch.ones(args.batch_size, states, dtype=torch.bool),
        value_mask=torch.ones(args.batch_size, states, dtype=torch.bool),
        indices=torch.arange(args.batch_size),
        importance_weights=torch.ones(args.batch_size),
        value_bootstrap_frames=bootstrap_frames,
        reanalysis_frames=reanalysis_frames,
        value_bootstrap_values=torch.zeros(args.batch_size, states),
        value_bootstrap_discounts=torch.ones(args.batch_size, states),
        value_bootstrap_mask=torch.ones(args.batch_size, states, dtype=torch.bool),
    )


def benchmark_depth(depth: int, args: argparse.Namespace, batch: ReplayBatch):
    representation = RepresentationNetwork(12)
    prediction = PredictionNetwork(args.action_space_size)
    dynamics = DynamicsNetwork(args.action_space_size)
    pipeline = ReanalysisPipeline(
        in_channels=12,
        action_space_size=args.action_space_size,
        search_config=SearchConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=5,
        ),
        policy_chunk_size=args.chunk_size,
        cache_targets=False,
        rng_seed=0,
        support_min=-300,
        support_max=300,
        precision=args.precision,
        search_threads=args.worker_threads,
        prefetch_batches=depth,
        timeout_seconds=600.0,
        target_update_interval=200,
        root_noise_total_steps=100_000,
    )
    try:
        pipeline.publish_weights(
            0,
            make_target_state(representation, prediction, dynamics),
            wait=True,
        )
        for _ in range(depth):
            pipeline.submit(batch)
        for _ in range(depth):
            pipeline.wait_next()

        submitted = completed = 0
        worker_times: list[float] = []
        request_latencies: list[float] = []
        started = perf_counter()
        while completed < args.requests:
            while submitted < args.requests and pipeline.needs_prefetch:
                pipeline.submit(batch)
                submitted += 1
            ready = pipeline.wait_next()
            worker_times.append(ready.worker_duration_ms)
            request_latencies.append(ready.queue_wait_ms)
            completed += 1
        elapsed = perf_counter() - started
        return {
            "prefetch_depth": depth,
            "elapsed_seconds": elapsed,
            "requests_per_second": args.requests / elapsed,
            "mean_worker_ms": mean(worker_times),
            "mean_request_latency_ms": mean(request_latencies),
            "peak_pending_batches": pipeline.max_observed_pending,
            "peak_pending_payload_bytes": pipeline.max_observed_pending_bytes,
        }
    finally:
        pipeline.close()


def main() -> None:
    args = parse_args()
    depths = tuple(int(value) for value in args.prefetch_depths.split(","))
    batch = make_batch(args)
    measurements = [benchmark_depth(depth, args, batch) for depth in depths]
    baseline = float(measurements[0]["requests_per_second"])
    for measurement in measurements:
        measurement["throughput_vs_first"] = (
            float(measurement["requests_per_second"]) / baseline
        )
    result = {
        "pipeline": "in_process_cpp_thread",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": args.batch_size,
        "unroll_steps": args.unroll_steps,
        "num_simulations": args.num_simulations,
        "chunk_size": args.chunk_size,
        "requests": args.requests,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
