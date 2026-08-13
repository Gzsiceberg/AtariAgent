#!/usr/bin/env python3
"""Benchmark the local Ray reanalysis pipeline with real Atari networks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import ray
import torch

from atariagent.models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTSConfig
from atariagent.training import (
    MuZeroTrainer,
    ReanalysisPipeline,
    create_reanalysis_actor,
    initialize_local_ray,
    make_target_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument("--actor-gpus", type=float, default=0.25)
    parser.add_argument("--actor-threads", type=int, default=4)
    return parser.parse_args()


def synthetic_batch(batch_size: int) -> ReplayBatch:
    generator = torch.Generator().manual_seed(0)
    frames = torch.randint(
        0,
        256,
        (batch_size, 5, 1, 96, 96),
        dtype=torch.uint8,
        generator=generator,
    )
    return ReplayBatch(
        frames=frames,
        actions=torch.randint(
            0, 3, (batch_size, 1, 1), generator=generator
        ),
        rewards=torch.zeros(batch_size, 1),
        policy_targets=torch.full((batch_size, 2, 3), 1.0 / 3.0),
        value_targets=torch.zeros(batch_size, 2),
        action_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        policy_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        value_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size),
        value_bootstrap_frames=frames.clone(),
        value_bootstrap_values=torch.zeros(batch_size, 2),
        value_bootstrap_discounts=torch.ones(batch_size, 2),
        value_bootstrap_mask=torch.ones(batch_size, 2, dtype=torch.bool),
    )


def main() -> None:
    args = parse_args()
    if args.warmup_updates < 0 or args.updates <= 0:
        raise ValueError("warmup must be non-negative and updates positive")
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2")

    owns_ray = initialize_local_ray()
    pipeline = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        representation = RepresentationNetwork(4).to(device)
        dynamics = DynamicsNetwork(action_space_size=3).to(device)
        prediction = PredictionNetwork(action_space_size=3).to(device)
        trainer = MuZeroTrainer(
            representation,
            dynamics,
            prediction,
            lr_warmup_steps=0,
            unroll_steps=1,
            lstm_horizon=1,
        )
        mcts_config = MCTSConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=1,
        )
        actor = create_reanalysis_actor(
            num_gpus=(args.actor_gpus if torch.cuda.is_available() else 0.0),
            num_cpus=args.actor_threads,
            mcts_threads=args.actor_threads,
            in_channels=4,
            action_space_size=3,
            mcts_config=mcts_config,
            policy_chunk_size=1024,
            cache_targets=False,
            rng_seed=0,
            support_min=-300,
            support_max=300,
            precision="fp32",
        )
        pipeline = ReanalysisPipeline(
            actor,
            prefetch_batches=2,
            timeout_seconds=600.0,
            max_weight_lag=200,
        )
        pipeline.publish_weights(
            0,
            make_target_state(representation, prediction, dynamics),
            wait=True,
        )
        cpu_batch = synthetic_batch(args.batch_size)
        global_update = 0

        def run_phase(count: int) -> tuple[float, list, list, list]:
            nonlocal global_update
            submitted = 0
            completed = 0
            actor_times: list[float] = []
            queue_times: list[float] = []
            actor_peaks: list[int] = []
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            while completed < count:
                while submitted < count and pipeline.needs_prefetch:
                    pipeline.submit(cpu_batch)
                    submitted += 1
                ready = pipeline.wait_next()
                batch = ready.batch.to(
                    device,
                    keep_indices_on_cpu=True,
                )
                trainer.train_step(batch)
                completed += 1
                global_update += 1
                actor_times.append(ready.actor_duration_ms)
                queue_times.append(ready.queue_wait_ms)
                actor_peaks.append(ready.peak_memory_bytes)
                if global_update % 200 == 0:
                    pipeline.publish_weights(
                        global_update,
                        make_target_state(
                            representation,
                            prediction,
                            dynamics,
                        ),
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return perf_counter() - started, actor_times, queue_times, actor_peaks

        if args.warmup_updates:
            run_phase(args.warmup_updates)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elapsed, actor_times, queue_times, actor_peaks = run_phase(args.updates)
        learner_peak = (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        )
        result = {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "batch_size": args.batch_size,
            "num_simulations": args.num_simulations,
            "actor_count": 1,
            "actor_threads": args.actor_threads,
            "policy_reanalysis_chunk_size": 1024,
            "prefetch_batches": 2,
            "warmup_updates": args.warmup_updates,
            "updates": args.updates,
            "elapsed_seconds": elapsed,
            "updates_per_second": args.updates / elapsed,
            "mean_actor_ms": mean(actor_times),
            "mean_queue_wait_ms": mean(queue_times),
            "peak_pending_batches": pipeline.max_observed_pending,
            "peak_pending_payload_bytes": (
                pipeline.max_observed_pending_bytes
            ),
            "actor_peak_memory_bytes": max(actor_peaks, default=0),
            "learner_peak_memory_bytes": learner_peak,
            "target_weight_version": pipeline.weight_version,
        }
        baseline_path = args.output.with_name("sync_reanalysis_baseline.json")
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            baseline_rate = float(baseline["updates_per_second"])
            result["sync_baseline_updates_per_second"] = baseline_rate
            result["speedup_percent"] = (
                result["updates_per_second"] / baseline_rate - 1.0
            ) * 100.0
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
