"""Checkpoint scheduling helpers for MuZero training."""

from __future__ import annotations

from pathlib import Path


def representative_checkpoint_updates(
    total_updates: int, count: int = 10
) -> tuple[int, ...]:
    """Return up to ``count`` evenly spaced updates, always including the end."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    return tuple(
        dict.fromkeys(
            (index * total_updates + count - 1) // count
            for index in range(1, count + 1)
        )
    )


def representative_checkpoint_path(latest_path: str | Path, update: int) -> Path:
    """Derive a numbered checkpoint name next to the latest checkpoint."""
    if update <= 0:
        raise ValueError("update must be positive")
    latest_path = Path(latest_path)
    stem = latest_path.stem
    if stem.endswith("_latest"):
        stem = stem[: -len("_latest")]
    return latest_path.with_name(f"{stem}_update_{update:08d}{latest_path.suffix}")


__all__ = [
    "representative_checkpoint_path",
    "representative_checkpoint_updates",
]
