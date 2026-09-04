#!/usr/bin/env python3
"""Benchmark the in-process native-thread reanalysis pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from atariagent.models import (
    ConsistencyNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from atariagent.replay_batch import ReplayBatch
from atariagent.search import SearchConfig
from atariagent.training import (
    ReanalysisPipeline,
    Trainer,
    make_target_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument("--worker-threads", type=int, default=4)
    return parser.parse_args()


def synthetic_batch(batch_size: int) -> ReplayBatch:
    generator = torch.Generator().manual_seed(0)
    frames = torch.randint(
        0, 256, (batch_size, 5, 1, 96, 96),
        dtype=torch.uint8, generator=generator,
    )
    return ReplayBatch(
        frames=frames,
        actions=torch.randint(0, 3, (batch_size, 1, 1), generator=generator),
        rewards=torch.zeros(batch_size, 1),
        policy_targets=torch.full((batch_size, 2, 3), 1.0 / 3.0),
        value_targets=torch.zeros(batch_size, 2),
        action_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        policy_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        value_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        indices=torch.arange(batch_size),
        importance_weights=torch.ones(batch_size),
        value_bootstrap_frames=frames,
        reanalysis_frames=frames,
        value_bootstrap_values=torch.zeros(batch_size, 2),
        value_bootstrap_discounts=torch.ones(batch_size, 2),
        value_bootstrap_mask=torch.ones(batch_size, 2, dtype=torch.bool),
    )


def main() -> None:
    args = parse_args()
    if args.warmup_updates < 0 or args.updates <= 0:
        raise ValueError("warmup must be non-negative and updates positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation = RepresentationNetwork(4).to(device)
    dynamics = DynamicsNetwork(3).to(device)
    prediction = PredictionNetwork(3).to(device)
    trainer = Trainer(
        representation, dynamics, prediction,
        consistency_network=ConsistencyNetwork().to(device),
        unroll_steps=1, lstm_horizon=1,
    )
    pipeline = ReanalysisPipeline(
        in_channels=4,
        action_space_size=3,
        search_config=SearchConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=1,
        ),
        policy_chunk_size=768,
        cache_targets=False,
        cache_target_ttl=0,
        rng_seed=0,
        support_min=-300,
        support_max=300,
        precision="fp32",
        search_threads=args.worker_threads,
        prefetch_batches=2,
        timeout_seconds=600.0,
        target_update_interval=200,
        device=device,
    )
    try:
        pipeline.publish_weights(
            0,
            make_target_state(representation, prediction, dynamics),
            wait=True,
        )
        cpu_batch = synthetic_batch(args.batch_size)
        global_update = 0

        def run_phase(count: int):
            nonlocal global_update
            submitted = completed = 0
            worker_times: list[float] = []
            queue_times: list[float] = []
            peaks: list[int] = []
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            while completed < count:
                while submitted < count and pipeline.needs_prefetch:
                    pipeline.submit(cpu_batch)
                    submitted += 1
                ready = pipeline.wait_next()
                trainer.train_step(
                    ready.batch.to(device, keep_indices_on_cpu=True)
                )
                completed += 1
                global_update += 1
                worker_times.append(ready.worker_duration_ms)
                queue_times.append(ready.queue_wait_ms)
                peaks.append(ready.peak_memory_bytes)
                if global_update % 200 == 0:
                    pipeline.publish_weights(
                        global_update,
                        make_target_state(representation, prediction, dynamics),
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return perf_counter() - started, worker_times, queue_times, peaks

        if args.warmup_updates:
            run_phase(args.warmup_updates)
        elapsed, worker_times, queue_times, peaks = run_phase(args.updates)
        result = {
            "pipeline": "in_process_cpp_thread",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "batch_size": args.batch_size,
            "num_simulations": args.num_simulations,
            "worker_threads": args.worker_threads,
            "prefetch_batches": 2,
            "warmup_updates": args.warmup_updates,
            "updates": args.updates,
            "elapsed_seconds": elapsed,
            "updates_per_second": args.updates / elapsed,
            "mean_worker_ms": mean(worker_times),
            "mean_queue_wait_ms": mean(queue_times),
            "peak_pending_batches": pipeline.max_observed_pending,
            "peak_pending_payload_bytes": pipeline.max_observed_pending_bytes,
            "worker_peak_memory_bytes": max(peaks, default=0),
            "target_weight_version": pipeline.weight_version,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
