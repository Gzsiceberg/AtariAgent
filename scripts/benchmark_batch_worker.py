#!/usr/bin/env python3
"""Benchmark synchronous replay preparation against threaded BatchWorker."""

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
from atariagent.replay import FIFOReplayBuffer
from atariagent.training import BatchWorker, Trainer
from benchmark_replay import make_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--trajectory-length", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--max-in-flight", type=int, default=3)
    parser.add_argument("--ready-prefetch", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_replay(args: argparse.Namespace, *, seed: int) -> FIFOReplayBuffer:
    replay = FIFOReplayBuffer(
        args.replay_size,
        unroll_steps=args.unroll_steps,
        td_steps=args.unroll_steps,
        discount=0.997,
        seed=seed,
    )
    remaining = args.replay_size
    block_id = 0
    while remaining:
        sampleable_steps = min(args.trajectory_length, remaining)
        replay.add(
            make_trajectory(
                sampleable_steps=sampleable_steps,
                lookahead_steps=args.unroll_steps,
                block_id=block_id,
                stack_size=4,
                channels=3,
                screen_size=96,
                action_space_size=18,
            )
        )
        remaining -= sampleable_steps
        block_id += 1
    return replay


def make_trainer(args: argparse.Namespace, device: torch.device) -> Trainer:
    torch.manual_seed(args.seed)
    representation = RepresentationNetwork(12).to(device)
    dynamics = DynamicsNetwork(18).to(device)
    prediction = PredictionNetwork(18).to(device)
    return Trainer(
        representation,
        dynamics,
        prediction,
        lr_warmup_steps=0,
        unroll_steps=args.unroll_steps,
        lstm_horizon=args.unroll_steps,
        precision=args.precision,
    )


def sample(replay: FIFOReplayBuffer, args: argparse.Namespace):
    return replay.sample(
        args.batch_size,
        include_value_bootstraps=False,
        pin_memory=True,
    )


def run_synchronous(
    replay: FIFOReplayBuffer,
    trainer: Trainer,
    args: argparse.Namespace,
    device: torch.device,
    updates: int,
) -> dict[str, float]:
    transfer_stream = torch.cuda.Stream(device=device)
    sample_times: list[float] = []

    def enqueue():
        started = perf_counter()
        cpu_batch = sample(replay, args)
        sample_times.append((perf_counter() - started) * 1_000.0)
        with torch.cuda.stream(transfer_stream):
            gpu_batch = cpu_batch.to(
                device, non_blocking=True, keep_indices_on_cpu=True
            )
            event = torch.cuda.Event()
            event.record(transfer_stream)
        return cpu_batch, gpu_batch, event

    started = perf_counter()
    current = enqueue()
    for step in range(updates):
        following = enqueue() if step + 1 < updates else None
        cpu_batch, gpu_batch, event = current
        stream = torch.cuda.current_stream(device)
        stream.wait_event(event)
        gpu_batch.record_stream(stream)
        metrics = trainer.train_step(gpu_batch)
        replay.update_priorities(gpu_batch.indices, metrics.priorities)
        del cpu_batch
        if following is not None:
            current = following
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "updates_per_second": updates / elapsed,
        "mean_sample_ms": mean(sample_times),
    }


def run_threaded(
    replay: FIFOReplayBuffer,
    trainer: Trainer,
    args: argparse.Namespace,
    device: torch.device,
    updates: int,
) -> dict[str, float]:
    sample_times: list[float] = []
    learner_wait_times: list[float] = []
    with BatchWorker(
        replay,
        batch_size=args.batch_size,
        device=device,
        max_in_flight=args.max_in_flight,
        ready_prefetch=args.ready_prefetch,
        timeout_seconds=120.0,
    ) as worker:
        started = perf_counter()
        worker.start(0, updates)
        for _ in range(updates):
            wait_started = perf_counter()
            ready = worker.next_ready()
            learner_wait_times.append((perf_counter() - wait_started) * 1_000.0)
            ready.wait_for_current_stream(device)
            metrics = trainer.train_step(ready.gpu_batch)
            worker.complete(ready, metrics.priorities)
            sample_times.append(ready.sample_duration_ms)
        worker.wait_idle()
        torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        max_observed = worker.max_observed_outstanding
    return {
        "elapsed_seconds": elapsed,
        "updates_per_second": updates / elapsed,
        "mean_sample_ms": mean(sample_times),
        "mean_learner_wait_ms": mean(learner_wait_times),
        "max_observed_outstanding": max_observed,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    for value, name in (
        (args.replay_size, "replay-size"),
        (args.trajectory_length, "trajectory-length"),
        (args.batch_size, "batch-size"),
        (args.unroll_steps, "unroll-steps"),
        (args.updates, "updates"),
        (args.max_in_flight, "max-in-flight"),
        (args.ready_prefetch, "ready-prefetch"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmup_updates < 0:
        raise ValueError("warmup-updates must be non-negative")
    if args.batch_size > args.replay_size:
        raise ValueError("batch-size must not exceed replay-size")

    device = torch.device("cuda")
    # Warm both execution paths and CUDA kernels outside measured phases.
    if args.warmup_updates:
        warm_replay = make_replay(args, seed=args.seed + 100)
        warm_trainer = make_trainer(args, device)
        run_threaded(
            warm_replay,
            warm_trainer,
            args,
            device,
            args.warmup_updates,
        )
        del warm_replay, warm_trainer
        torch.cuda.empty_cache()

    sync_replay = make_replay(args, seed=args.seed)
    sync_trainer = make_trainer(args, device)
    synchronous = run_synchronous(
        sync_replay, sync_trainer, args, device, args.updates
    )
    del sync_replay, sync_trainer
    torch.cuda.empty_cache()

    threaded_replay = make_replay(args, seed=args.seed)
    threaded_trainer = make_trainer(args, device)
    threaded = run_threaded(
        threaded_replay, threaded_trainer, args, device, args.updates
    )
    speedup = (
        threaded["updates_per_second"]
        / synchronous["updates_per_second"]
    )
    result = {
        "device": torch.cuda.get_device_name(device),
        "precision": args.precision,
        "replay_size": args.replay_size,
        "batch_size": args.batch_size,
        "unroll_steps": args.unroll_steps,
        "updates": args.updates,
        "max_in_flight": args.max_in_flight,
        "ready_prefetch": args.ready_prefetch,
        "synchronous": synchronous,
        "threaded": threaded,
        "threaded_speedup": speedup,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
