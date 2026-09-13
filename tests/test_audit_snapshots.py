"""Replay audit counts, coverage, and content fingerprints."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "audit_snapshots", Path(__file__).resolve().parents[1] / "scripts/audit_snapshots.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot(*, truncated=False, inserted=2, version=1):
    block = {
        "environment_index": 0, "episode_id": 1, "block_id": 0,
        "terminated": False, "truncated": truncated,
        "actions": np.array([0]), "rewards": np.array([1.0]),
    }
    return {
        "snapshot_type": "atariagent_pre_final", "snapshot_version": version,
        "update": 100, "config": {},
        "replay": {
            "trajectories": [block, {**block, "block_id": 1}],
            "transition_count": 2, "next_transition_id": inserted,
        },
        "representation": {}, "dynamics": {}, "prediction": {},
    }


@pytest.mark.parametrize("version", [1, 2])
def test_zero_without_eviction(version):
    report = module.audit_snapshot(snapshot(version=version))
    assert report["truncated_blocks"] == 0
    assert report["covers_all_inserted_transitions"]
    assert report["evicted_transitions"] == 0


def test_overlap_is_not_double_counted_as_episode():
    report = module.audit_snapshot(snapshot(truncated=True))
    assert report["truncated_blocks"] == 2
    assert report["truncated_episode_keys"] == 1


def test_eviction_limits_coverage():
    report = module.audit_snapshot(snapshot(inserted=5))
    assert not report["covers_all_inserted_transitions"]
    assert report["evicted_transitions"] == 3


def test_digest_detects_different_training_data():
    original = snapshot()
    changed = snapshot()
    changed["replay"]["trajectories"][0]["actions"] = np.array([1])
    assert module.audit_snapshot(original)["action_reward_digest"] != (
        module.audit_snapshot(changed)["action_reward_digest"]
    )


def test_rejects_non_replay_checkpoint():
    with pytest.raises(ValueError, match="not an AtariAgent"):
        module.audit_snapshot({})
