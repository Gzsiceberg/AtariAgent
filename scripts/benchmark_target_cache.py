#!/usr/bin/env python3
"""Benchmark driver-side target-version value and policy caching."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import ray
import torch

from atariagent.models import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from atariagent.search import MCTSConfig
from atariagent.training import (
    MuZeroTrainer,
    ReanalysisPipeline,
    create_reanalysis_actor,
    initialize_local_ray,
    make_target_state,
)
from atariagent.typecheck import set_runtime_typechecking
from benchmark_reanalysis_pool import make_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--replay-size", type=int, default=10_000)
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


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.replay_size <= args.unroll_steps:
        raise ValueError("updates must be positive and replay must cover the unroll")
    if args.batch_size <= 1 or args.batch_size > args.replay_size:
        raise ValueError("batch size must be in [2, replay size]")

    set_runtime_typechecking(False)
    owns_ray = initialize_local_ray()
    pipeline = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_batch = make_batch(args)
        rng = np.random.default_rng(0)
        maximum_start = args.replay_size - args.unroll_steps
        batches = tuple(
            replace(
                base_batch,
                indices=torch.from_numpy(
                    rng.choice(
                        maximum_start,
                        args.batch_size,
                        replace=False,
                    )
                ).long(),
            )
            for _ in range(args.updates)
        )

        representation = RepresentationNetwork(12).to(device)
        prediction = PredictionNetwork(args.action_space_size).to(device)
        dynamics = DynamicsNetwork(args.action_space_size).to(device)
        trainer = MuZeroTrainer(
            representation,
            dynamics,
            prediction,
            unroll_steps=args.unroll_steps,
            lstm_horizon=5,
            precision=args.precision,
            compile_model=True,
            lr_warmup_steps=0,
        )
        # Compile the learner before timing without warming the policy cache.
        trainer.train_step(
            base_batch.to(device, keep_indices_on_cpu=True)
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        actor = create_reanalysis_actor(
            num_gpus=(args.actor_gpus if device.type == "cuda" else 0.0),
            num_cpus=args.actor_threads,
            mcts_threads=args.actor_threads,
            in_channels=12,
            action_space_size=args.action_space_size,
            mcts_config=MCTSConfig(
                num_simulations=args.num_simulations,
                value_prefix_horizon=5,
            ),
            policy_chunk_size=args.chunk_size,
            cache_targets=True,
            rng_seed=0,
            support_min=-300,
            support_max=300,
            precision=args.precision,
        )
        pipeline = ReanalysisPipeline(
            actor,
            prefetch_batches=2,
            timeout_seconds=600.0,
            max_weight_lag=0,
        )
        pipeline.publish_weights(
            0,
            make_target_state(representation, prediction, dynamics),
            wait=True,
        )

        submitted = 0
        completed = 0
        requested_roots = 0
        searched_policy_roots = 0
        actor_time_ms = 0.0
        started = perf_counter()
        while completed < args.updates:
            while submitted < args.updates and pipeline.needs_prefetch:
                pipeline.submit(batches[submitted])
                submitted += 1
            ready = pipeline.wait_next()
            requested_roots += ready.policy_roots_requested
            searched_policy_roots += ready.policy_roots_searched
            actor_time_ms += ready.actor_duration_ms
            cpu_batch = (
                ready.batch.pin_memory()
                if device.type == "cuda"
                else ready.batch
            )
            batch = cpu_batch.to(
                device,
                non_blocking=device.type == "cuda",
                keep_indices_on_cpu=True,
            )
            trainer.train_step(batch)
            completed += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started

        result = {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "updates": args.updates,
            "replay_size": args.replay_size,
            "batch_size": args.batch_size,
            "unroll_steps": args.unroll_steps,
            "num_simulations": args.num_simulations,
            "actor_count": 1,
            "elapsed_seconds": elapsed,
            "updates_per_second": args.updates / elapsed,
            "mean_actor_ms": actor_time_ms / args.updates,
            "policy_roots_requested": requested_roots,
            "policy_roots_searched": searched_policy_roots,
            "policy_search_reduction": (
                requested_roots / searched_policy_roots
            ),
            "cache_size": pipeline.cache_size,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if pipeline is not None:
            pipeline.close()
        if owns_ray and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
