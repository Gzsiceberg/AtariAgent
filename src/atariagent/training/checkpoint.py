"""Checkpoint scheduling helpers for AtariAgent training."""

from __future__ import annotations

from pathlib import Path


def representative_checkpoint_updates(
    collection_updates: int,
    final_updates: int,
    *,
    collection_interval: int = 10_000,
    final_interval: int = 5_000,
) -> tuple[int, ...]:
    """Schedule checkpoints independently for collection and final phases.

    Each non-empty phase includes its endpoint even when its length is not an
    exact multiple of the configured interval.
    """
    if collection_updates <= 0:
        raise ValueError("collection_updates must be positive")
    if final_updates < 0:
        raise ValueError("final_updates must be non-negative")
    if collection_interval <= 0:
        raise ValueError("collection_interval must be positive")
    if final_interval <= 0:
        raise ValueError("final_interval must be positive")

    collection = list(
        range(collection_interval, collection_updates + 1, collection_interval)
    )
    if not collection or collection[-1] != collection_updates:
        collection.append(collection_updates)

    total_updates = collection_updates + final_updates
    final = list(
        range(
            collection_updates + final_interval,
            total_updates + 1,
            final_interval,
        )
    )
    if final_updates > 0 and (not final or final[-1] != total_updates):
        final.append(total_updates)
    return tuple(collection + final)


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
