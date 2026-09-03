"""Benchmark direct and MCTS TD-endpoint reanalysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from atariagent.models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from atariagent.replay_batch import ReplayBatch
from atariagent.search import SearchConfig
from atariagent.training import ReanalysisPipeline, make_target_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=5)
    parser.add_argument("--action-space-size", type=int, default=18)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=768)
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_batch(args: argparse.Namespace, *, reuse_overlap: bool) -> ReplayBatch:
    states = args.unroll_steps + 1
    stack_size = 4
    frame_count = stack_size + args.unroll_steps
    bootstrap_offset = args.td_steps if reuse_overlap else states
    # Consolidated bootstrap frames are represented as a suffix. The forced
    # no-overlap baseline therefore transfers one extra frame per sample.
    combined_count = max(frame_count, bootstrap_offset + frame_count)
    reanalysis_frames = torch.zeros(
        args.batch_size,
        combined_count,
        3,
        96,
        96,
        dtype=torch.uint8,
    )
    frames = reanalysis_frames[:, :frame_count]
    bootstrap_frames = reanalysis_frames[
        :, bootstrap_offset : bootstrap_offset + frame_count
    ]
    return ReplayBatch(
        frames=frames,
        actions=torch.zeros(
            args.batch_size, args.unroll_steps, 1, dtype=torch.long
        ),
        rewards=torch.zeros(args.batch_size, args.unroll_steps),
        policy_targets=torch.full(
            (args.batch_size, states, args.action_space_size),
            1.0 / args.action_space_size,
        ),
        value_targets=torch.zeros(args.batch_size, states),
        action_mask=torch.ones(
            args.batch_size, args.unroll_steps, dtype=torch.bool
        ),
        policy_mask=torch.ones(args.batch_size, states, dtype=torch.bool),
        value_mask=torch.ones(args.batch_size, states, dtype=torch.bool),
        indices=torch.arange(args.batch_size),
        importance_weights=torch.ones(args.batch_size),
        value_bootstrap_frames=bootstrap_frames,
        value_bootstrap_values=torch.zeros(args.batch_size, states),
        value_bootstrap_discounts=torch.ones(args.batch_size, states),
        value_bootstrap_mask=torch.ones(
            args.batch_size, states, dtype=torch.bool
        ),
        mcts_bootstrap_mask=torch.ones(
            args.batch_size, states, dtype=torch.bool
        ),
        reanalysis_frames=reanalysis_frames,
        reanalysis_state_ids=(
            torch.arange(args.batch_size)[:, None] * states
            + torch.arange(states)[None, :]
        ),
    )


def run_once(
    args: argparse.Namespace,
    *,
    batch: ReplayBatch,
    target_state: dict[str, torch.Tensor],
    use_mcts_bootstrap: bool,
) -> dict[str, float | int]:
    pipeline = ReanalysisPipeline(
        in_channels=12,
        action_space_size=args.action_space_size,
        search_config=SearchConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=5,
        ),
        policy_chunk_size=args.chunk_size,
        cache_targets=False,
        cache_target_ttl=0,
        rng_seed=0,
        support_min=-300,
        support_max=300,
        precision=args.precision,
        search_threads=args.worker_threads,
        prefetch_batches=2,
        timeout_seconds=600.0,
        target_update_interval=1_000,
        mcts_bootstrap_start_step=0 if use_mcts_bootstrap else None,
    )
    try:
        pipeline.publish_weights(0, target_state, wait=True)

        def run_requests(count: int) -> tuple[list[float], int, int]:
            submitted = 0
            completed = 0
            worker_times: list[float] = []
            policy_roots = 0
            bootstrap_roots = 0
            while completed < count:
                while submitted < count and pipeline.needs_prefetch:
                    pipeline.submit(batch, trained_steps=0)
                    submitted += 1
                ready = pipeline.wait_next()
                worker_times.append(ready.worker_duration_ms)
                policy_roots += ready.policy_roots_searched
                bootstrap_roots += ready.bootstrap_roots_searched
                completed += 1
            return worker_times, policy_roots, bootstrap_roots

        run_requests(args.warmup_requests)
        torch.cuda.synchronize()
        started = perf_counter()
        worker_times, policy_roots, bootstrap_roots = run_requests(args.requests)
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        return {
            "elapsed_seconds": elapsed,
            "requests_per_second": args.requests / elapsed,
            "mean_worker_ms": mean(worker_times),
            "policy_roots_searched": policy_roots,
            "bootstrap_roots_searched": bootstrap_roots,
        }
    finally:
        pipeline.close()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.requests <= 0 or args.warmup_requests < 0 or args.repetitions <= 0:
        raise ValueError("request counts and repetitions are invalid")

    representation = RepresentationNetwork(12)
    prediction = PredictionNetwork(args.action_space_size)
    dynamics = DynamicsNetwork(args.action_space_size)
    target_state = make_target_state(representation, prediction, dynamics)
    overlap_batch = make_batch(args, reuse_overlap=True)
    no_overlap_batch = make_batch(args, reuse_overlap=False)
    stale_count = round(0.95 * args.batch_size)
    mixed_mask = torch.zeros_like(overlap_batch.mcts_bootstrap_mask)
    mixed_mask[:stale_count] = True
    mixed_batch = replace(overlap_batch, mcts_bootstrap_mask=mixed_mask)
    modes = {
        "direct_bootstrap": (overlap_batch, False),
        "mcts_bootstrap_95pct_stale": (mixed_batch, True),
        "mcts_bootstrap_reuse": (overlap_batch, True),
        "mcts_bootstrap_no_reuse": (no_overlap_batch, True),
    }

    samples: dict[str, list[dict[str, float | int]]] = {
        mode: [] for mode in modes
    }
    mode_names = tuple(modes)
    for repetition in range(args.repetitions):
        # Rotate execution order to reduce thermal/order bias.
        ordered_modes = (
            mode_names[repetition % len(mode_names) :]
            + mode_names[: repetition % len(mode_names)]
        )
        for mode in ordered_modes:
            batch, use_mcts = modes[mode]
            samples[mode].append(
                run_once(
                    args,
                    batch=batch,
                    target_state=target_state,
                    use_mcts_bootstrap=use_mcts,
                )
            )

    summary: dict[str, dict[str, float | int]] = {}
    for mode, measurements in samples.items():
        summary[mode] = {
            "median_requests_per_second": median(
                float(item["requests_per_second"]) for item in measurements
            ),
            "median_worker_ms": median(
                float(item["mean_worker_ms"]) for item in measurements
            ),
            "policy_roots_per_request": int(
                measurements[0]["policy_roots_searched"]
            )
            // args.requests,
            "bootstrap_roots_per_request": int(
                measurements[0]["bootstrap_roots_searched"]
            )
            // args.requests,
        }

    direct_ms = float(summary["direct_bootstrap"]["median_worker_ms"])
    mixed_ms = float(
        summary["mcts_bootstrap_95pct_stale"]["median_worker_ms"]
    )
    reuse_ms = float(summary["mcts_bootstrap_reuse"]["median_worker_ms"])
    no_reuse_ms = float(summary["mcts_bootstrap_no_reuse"]["median_worker_ms"])
    result = {
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "unroll_steps": args.unroll_steps,
        "td_steps": args.td_steps,
        "num_simulations": args.num_simulations,
        "requests_per_repetition": args.requests,
        "repetitions": args.repetitions,
        "summary": summary,
        "mcts_95pct_worker_slowdown_vs_direct": mixed_ms / direct_ms,
        "mcts_reuse_worker_slowdown_vs_direct": reuse_ms / direct_ms,
        "reuse_worker_speedup_vs_no_reuse": no_reuse_ms / reuse_ms,
        "raw_measurements": samples,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
