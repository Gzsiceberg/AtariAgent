#!/usr/bin/env python3
"""Profile the native PUCT and Gumbel tree implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
from atariagent.search._tree_search_native import BatchTree, set_num_threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=int, default=8192)
    parser.add_argument("--actions", type=int, default=18)
    parser.add_argument("--puct-simulations", type=int, default=50)
    parser.add_argument("--gumbel-simulations", type=int, default=16)
    parser.add_argument("--num-top-actions", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": samples,
        "mean_ms": mean(samples),
        "median_ms": median(samples),
    }


def run_algorithm(
    algorithm: str,
    *,
    roots: int,
    actions: int,
    simulations: int,
    num_top_actions: int,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float | list[float]]]:
    priors = np.full((roots, actions), 1.0 / actions, dtype=np.float32)
    values = np.zeros(roots, dtype=np.float32)
    logits = np.zeros((roots, actions), dtype=np.float32)
    phases = {name: [] for name in ("construction", "traversal", "expansion", "output", "total")}

    for iteration in range(warmup + iterations):
        total_started = perf_counter()
        started = perf_counter()
        tree = BatchTree(
            priors,
            values,
            values,
            simulations,
            0.997,
            5,
            0.01,
            iteration,
            True,
            algorithm,
            num_top_actions,
            50.0,
            0.1,
            False,
        )
        construction = perf_counter() - started
        traversal = 0.0
        expansion = 0.0
        for simulation in range(simulations):
            started = perf_counter()
            tree.traverse_arrays(19652.0, 1.25)
            traversal += perf_counter() - started
            started = perf_counter()
            tree.expand_and_back_up_arrays(
                simulation + 1,
                values,
                values,
                logits,
            )
            expansion += perf_counter() - started
        started = perf_counter()
        tree.visit_counts_array()
        tree.root_values_array()
        if algorithm == "gumbel":
            tree.policy_array()
            tree.selected_actions_array()
        output = perf_counter() - started
        total = perf_counter() - total_started
        if iteration >= warmup:
            for name, duration in (
                ("construction", construction),
                ("traversal", traversal),
                ("expansion", expansion),
                ("output", output),
                ("total", total),
            ):
                phases[name].append(duration * 1000.0)

    return {name: summarize(samples) for name, samples in phases.items()}


def main() -> None:
    args = parse_args()
    if min(
        args.roots,
        args.actions,
        args.puct_simulations,
        args.gumbel_simulations,
        args.num_top_actions,
        args.threads,
        args.iterations,
    ) <= 0 or args.warmup < 0:
        raise ValueError("benchmark sizes must be positive")
    set_num_threads(args.threads)
    simulation_counts = {
        "puct": args.puct_simulations,
        "gumbel": args.gumbel_simulations,
    }
    algorithms = {
        algorithm: run_algorithm(
            algorithm,
            roots=args.roots,
            actions=args.actions,
            simulations=simulation_counts[algorithm],
            num_top_actions=args.num_top_actions,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for algorithm in ("puct", "gumbel")
    }
    puct = algorithms["puct"]["total"]["mean_ms"]
    gumbel = algorithms["gumbel"]["total"]["mean_ms"]
    assert isinstance(puct, float) and isinstance(gumbel, float)
    result = {
        "roots": args.roots,
        "actions": args.actions,
        "simulation_counts": simulation_counts,
        "num_top_actions": args.num_top_actions,
        "threads": args.threads,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "algorithms": algorithms,
        "gumbel_vs_puct_speedup": puct / gumbel,
        "gumbel_time_reduction_percent": (1.0 - gumbel / puct) * 100.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
