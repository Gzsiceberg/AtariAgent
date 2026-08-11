"""Opt-in learner timing aggregation.

Profiling deliberately synchronizes CUDA at section boundaries. It must stay
disabled for ordinary throughput runs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from statistics import mean

import torch


@dataclass(frozen=True, slots=True)
class LearnerTimingSummary:
    """Steady-state timing and memory statistics for a reporting window."""

    updates: int
    updates_per_second: float
    samples_per_second: float
    mean_update_ms: float
    p95_update_ms: float
    mean_sections_ms: dict[str, float]
    peak_allocated_mib: float
    peak_reserved_mib: float


class LearnerProfiler:
    """Aggregate synchronized update timings after a fixed warmup."""

    def __init__(
        self,
        *,
        warmup_steps: int,
        report_every: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if report_every <= 0:
            raise ValueError("report_every must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.warmup_steps = warmup_steps
        self.report_every = report_every
        self.batch_size = batch_size
        self.device = device
        self._updates_seen = 0
        self._sections: dict[str, list[float]] = defaultdict(list)

    def record(self, timings_ms: Mapping[str, float]) -> LearnerTimingSummary | None:
        """Record an update and return a summary at reporting boundaries."""
        self._updates_seen += 1
        if self._updates_seen == self.warmup_steps and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if self._updates_seen <= self.warmup_steps:
            return None

        for name, duration in timings_ms.items():
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError(f"invalid {name} profiling duration")
            self._sections[name].append(float(duration))

        update_durations = self._sections.get("update_total", [])
        if len(update_durations) < self.report_every:
            return None

        ordered = sorted(update_durations)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        total_seconds = max(sum(update_durations) / 1_000.0, 1e-12)
        updates_per_second = len(update_durations) / total_seconds
        if self.device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            peak_allocated = 0
            peak_reserved = 0
        summary = LearnerTimingSummary(
            updates=len(update_durations),
            updates_per_second=updates_per_second,
            samples_per_second=updates_per_second * self.batch_size,
            mean_update_ms=mean(update_durations),
            p95_update_ms=ordered[p95_index],
            mean_sections_ms={
                name: mean(durations)
                for name, durations in self._sections.items()
                if durations
            },
            peak_allocated_mib=peak_allocated / (1024**2),
            peak_reserved_mib=peak_reserved / (1024**2),
        )
        self._sections.clear()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return summary


def synchronize_for_profiling(device: torch.device, enabled: bool) -> None:
    """Synchronize a CUDA device only when detailed profiling is enabled."""
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = [
    "LearnerProfiler",
    "LearnerTimingSummary",
    "synchronize_for_profiling",
]
