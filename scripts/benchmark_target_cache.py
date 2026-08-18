#!/usr/bin/env python3
"""Benchmark target caching in the native reanalysis pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from benchmark_reanalysis_pool import make_batch

from atariagent.models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from atariagent.search import SearchConfig
from atariagent.training import (
    ReanalysisPipeline,
    Trainer,
    make_target_state,
)
from atariagent.typecheck import set_runtime_typechecking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--action-space-size", type=int, default=18)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=768)
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--publication-repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def cache_hit_buckets(
    samples: list[tuple[float, float]],
) -> dict[str, dict[str, float | int]]:
    boundaries = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    result: dict[str, dict[str, float | int]] = {}
    for lower, upper in pairwise(boundaries):
        durations = [
            duration
            for hit_fraction, duration in samples
            if lower <= hit_fraction < upper
        ]
        if not durations:
            continue
        label = f"{lower:.2f}-{min(upper, 1.0):.2f}"
        result[label] = {"requests": len(durations), **distribution(durations)}
    fully_cached = [
        duration for hit_fraction, duration in samples if hit_fraction == 1.0
    ]
    if fully_cached:
        result["1.00"] = {
            "requests": len(fully_cached),
            **distribution(fully_cached),
        }
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.publication_repetitions <= 0:
        raise ValueError("updates and publication repetitions must be positive")
    set_runtime_typechecking(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_batch = make_batch(args)
    rng = np.random.default_rng(0)
    maximum_start = args.replay_size - args.unroll_steps
    batches = tuple(
        replace(
            base_batch,
            indices=torch.from_numpy(
                rng.choice(maximum_start, args.batch_size, replace=False)
            ).long(),
        )
        for _ in range(args.updates)
    )
    representation = RepresentationNetwork(12).to(device)
    prediction = PredictionNetwork(args.action_space_size).to(device)
    dynamics = DynamicsNetwork(args.action_space_size).to(device)
    trainer = Trainer(
        representation, dynamics, prediction,
        unroll_steps=args.unroll_steps, lstm_horizon=5,
        precision=args.precision, compile_model=True,
    )
    trainer.train_step(
        base_batch.without_reanalysis_metadata().to(
            device, keep_indices_on_cpu=True
        )
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pipeline = ReanalysisPipeline(
        in_channels=12,
        action_space_size=args.action_space_size,
        search_config=SearchConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=5,
        ),
        policy_chunk_size=args.chunk_size,
        cache_targets=True,
        rng_seed=0,
        support_min=-300,
        support_max=300,
        precision=args.precision,
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
        submitted = completed = requested = searched = 0
        worker_ms = 0.0
        cache_samples: list[tuple[float, float]] = []
        started = perf_counter()
        while completed < args.updates:
            while submitted < args.updates and pipeline.needs_prefetch:
                pipeline.submit(batches[submitted])
                submitted += 1
            ready = pipeline.wait_next()
            requested += ready.policy_roots_requested
            searched += ready.policy_roots_searched
            worker_ms += ready.worker_duration_ms
            hit_fraction = 1.0 - (
                ready.policy_roots_searched / ready.policy_roots_requested
            )
            cache_samples.append((hit_fraction, ready.worker_duration_ms))
            learner_batch = ready.batch.without_reanalysis_metadata()
            trainer.train_step(
                learner_batch.to(device, keep_indices_on_cpu=True)
            )
            completed += 1
        synchronize(device)
        elapsed = perf_counter() - started

        cache_size_before_publication = pipeline.cache_size
        snapshot_ms: list[float] = []
        publication_ms: list[float] = []
        for repetition in range(args.publication_repetitions):
            synchronize(device)
            snapshot_started = perf_counter()
            state = make_target_state(representation, prediction, dynamics)
            synchronize(device)
            snapshot_ms.append((perf_counter() - snapshot_started) * 1_000.0)

            publication_started = perf_counter()
            pipeline.publish_weights(args.updates + repetition + 1, state)
            synchronize(device)
            publication_ms.append(
                (perf_counter() - publication_started) * 1_000.0
            )

        state_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in state.values()
        )
        result = {
            "pipeline": "in_process_cpp_thread",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "updates": args.updates,
            "replay_size": args.replay_size,
            "batch_size": args.batch_size,
            "unroll_steps": args.unroll_steps,
            "num_simulations": args.num_simulations,
            "elapsed_seconds": elapsed,
            "updates_per_second": args.updates / elapsed,
            "mean_worker_ms": worker_ms / args.updates,
            "policy_roots_requested": requested,
            "policy_roots_searched": searched,
            "policy_search_reduction": requested / searched,
            "cache_size": cache_size_before_publication,
            "cache_size_before_publication_profile": (
                cache_size_before_publication
            ),
            "cache_worker_ms_by_effective_hit_fraction": cache_hit_buckets(
                cache_samples
            ),
            "target_state_bytes": state_bytes,
            "target_snapshot_gpu_to_cpu": distribution(snapshot_ms),
            "target_publication_cpu_to_gpu": distribution(publication_ms),
            "target_snapshot_and_publication_mean_ms": float(
                np.mean(snapshot_ms) + np.mean(publication_ms)
            ),
            "publication_repetitions": args.publication_repetitions,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
