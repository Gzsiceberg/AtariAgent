"""Audit trusted pre-final replay snapshots without initializing training or CUDA.

uv run python scripts/audit_snapshots.py root=runs/tsp_sweep/20260907_211119

Counts refer to retained replay blocks, not W&B's boundary-owning full episodes.
A zero count verifies the entire inserted replay history only if no transitions
were evicted. It cannot cover trajectories never inserted into replay.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig


def fingerprint(value: object) -> str:
    """Stable content digest (independent of pickle storage IDs)."""
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            update(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                update(entry)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def audit_snapshot(snapshot: Mapping) -> dict:
    if snapshot.get("snapshot_type") != "atariagent_pre_final":
        raise ValueError("not an AtariAgent pre-final snapshot")
    # Historical v2 adds bootstrap-target networks; replay flags are unchanged.
    if snapshot.get("snapshot_version") not in (1, 2):
        raise ValueError("unsupported snapshot version")
    replay = snapshot["replay"]
    trajectories = replay["trajectories"]
    # Flags may occur in overlapping lookahead blocks: do not label these
    # counts as the logger's completed-episode count.
    truncated = [t for t in trajectories if t["truncated"]]
    config = snapshot["config"]
    retained = int(replay["transition_count"])
    inserted = int(replay["next_transition_id"])
    return {
        "update": snapshot["update"],
        "retained_transitions": retained,
        "inserted_transitions": inserted,
        "evicted_transitions": inserted - retained,
        "covers_all_inserted_transitions": inserted == retained,
        "blocks": len(trajectories),
        "terminated_blocks": sum(bool(t["terminated"]) for t in trajectories),
        "truncated_blocks": len(truncated),
        "truncated_episode_keys": len({
            (t["environment_index"], t["episode_id"]) for t in truncated
        }),
        "truncated_block_ids": [
            [t["environment_index"], t["episode_id"], t["block_id"]]
            for t in truncated
        ],
        "config": config,
        "action_reward_digest": fingerprint([
            {k: t[k] for k in (
                "environment_index", "episode_id", "block_id", "actions", "rewards"
            )} for t in trajectories
        ]),
        "network_digest": fingerprint({
            k: snapshot[k] for k in ("representation", "dynamics", "prediction")
        }),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="audit_snapshots")
def main(config: DictConfig) -> None:
    root = Path(config.root)
    paths = [root] if root.is_file() else sorted(root.rglob(config.snapshot_name))
    if not paths:
        raise FileNotFoundError(f"no {config.snapshot_name} snapshots under {root}")
    reports = []
    for path in paths:
        print(f"Loading {path}", flush=True)
        # Never load untrusted files: replay snapshots require Python pickle.
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        report = {"path": str(path), **audit_snapshot(snapshot)}
        del snapshot  # Only one multi-GB replay snapshot in memory at a time.
        evaluation_path = path.parent.parent / "evaluations/agent_evaluations.json"
        if evaluation_path.is_file():
            report["evaluations"] = json.loads(evaluation_path.read_text())["evaluations"]
        reports.append(report)
        print(
            f"  truncated blocks={report['truncated_blocks']}, "
            f"episode keys={report['truncated_episode_keys']}; "
            f"retained/inserted={report['retained_transitions']}/"
            f"{report['inserted_transitions']}; "
            f"evicted={report['evicted_transitions']}",
            flush=True,
        )
    if root.is_dir():
        for job in sorted(root.rglob("job.sh")):
            if not (job.parent / "checkpoints" / config.snapshot_name).is_file():
                print(f"No replay snapshot: {job.parent}")
    print("Counts audit retained replay, not the cumulative W&B episode counter.")
    if config.output_path is not None:
        output = Path(config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, indent=2) + "\n")
        print(f"Report: {output}")


if __name__ == "__main__":
    main()
