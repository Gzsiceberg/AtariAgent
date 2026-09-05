#!/usr/bin/env python3
"""Summarize saved systems benchmark windows without accessing the GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main():
    root = Path(sys.argv[1])
    result = {}
    for directory in sorted(root.iterdir()):
        if not (directory / "summary.json").is_file():
            continue  # Exclude incomplete/pilot runs.
        windows = [
            json.loads(p.read_text()) for p in sorted(directory.glob("repeat_*.json"))
        ]
        samples = {}
        for window in windows:
            for key, values in window["samples"].items():
                samples.setdefault(key, []).extend(values)
        summary = json.loads((directory / "summary.json").read_text())
        buckets = {}
        requested = samples.get("policy_roots_requested", [])
        searched = samples.get("policy_roots_searched", [])
        times = samples.get("worker_duration_ms", [])
        for lower, upper in (
            (0, 0.5),
            (0.5, 0.75),
            (0.75, 0.9),
            (0.9, 0.99),
            (0.99, 1.00001),
        ):
            selected = [
                ms
                for n, miss, ms in (
                    zip(requested, searched, times, strict=True) if times else ()
                )
                if n and lower <= 1 - miss / n < upper
            ]
            if selected:
                buckets[f"{lower}-{min(upper, 1)}"] = {
                    "count": len(selected),
                    "mean_ms": float(np.mean(selected)),
                    "p50_ms": float(np.median(selected)),
                    "p95_ms": float(np.percentile(selected, 95)),
                }
        result[directory.name] = {
            "throughputs": [w["updates_per_second"] for w in windows],
            "pooled_updates_per_second": sum(
                len(w["samples"]["learner_enqueue_cpu_ms"]) for w in windows
            )
            / sum(w["elapsed_seconds"] for w in windows),
            "latencies": {
                k: {
                    "mean": float(np.mean(v)),
                    "p50": float(np.median(v)),
                    "p95": float(np.percentile(v, 95)),
                    "max": float(max(v)),
                }
                for k, v in samples.items()
                if v
            },
            "effective_policy_hit_fraction": 1 - sum(searched) / sum(requested)
            if sum(requested)
            else None,
            "worker_ms_by_effective_hit_fraction": buckets,
            "warmup_pipeline_seconds": summary["warmup_pipeline_seconds"],
            "compile_and_learner_warmup_ms": summary["setup_samples"][
                "learner_compile_and_warmup_ms"
            ],
            "isolated": {
                k: {
                    "mean": float(np.mean(v)),
                    "p50": float(np.median(v)),
                    "p95": float(np.percentile(v, 95)),
                }
                for k, v in summary["isolated_samples"].items()
            },
        }
    (root / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
