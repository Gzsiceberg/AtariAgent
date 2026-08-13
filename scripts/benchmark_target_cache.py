#!/usr/bin/env python3
"""Benchmark target caching in the native reanalysis pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from atariagent.models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from atariagent.search import MCTSConfig
from atariagent.training import (
    MuZeroTrainer,
    ReanalysisPipeline,
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
    parser.add_argument("--chunk-size", type=int, default=768)
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    trainer = MuZeroTrainer(
        representation, dynamics, prediction,
        unroll_steps=args.unroll_steps, lstm_horizon=5,
        precision=args.precision, compile_model=True, lr_warmup_steps=0,
    )
    trainer.train_step(base_batch.to(device, keep_indices_on_cpu=True))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pipeline = ReanalysisPipeline(
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
        mcts_threads=args.worker_threads,
        prefetch_batches=2,
        timeout_seconds=600.0,
        max_weight_lag=0,
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
        started = perf_counter()
        while completed < args.updates:
            while submitted < args.updates and pipeline.needs_prefetch:
                pipeline.submit(batches[submitted])
                submitted += 1
            ready = pipeline.wait_next()
            requested += ready.policy_roots_requested
            searched += ready.policy_roots_searched
            worker_ms += ready.worker_duration_ms
            learner_batch = ready.batch.without_value_bootstraps()
            trainer.train_step(
                learner_batch.to(device, keep_indices_on_cpu=True)
            )
            completed += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
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
            "cache_size": pipeline.cache_size,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
