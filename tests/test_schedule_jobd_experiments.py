"""Exercise shared-queue submission without contacting jobd or training."""

import json
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/schedule_jobd_experiments.sh"


@pytest.fixture
def scheduler(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    jobd = binaries / "jobd"
    jobd.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    jobd.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "CALLS": str(calls),
        "JOBD_API_KEY": "controller-secret",
        "WANDB_API_KEY": "wandb-secret",
        "RUN_ID": "test",
        "RUN_ROOT": "runs/test",
        "WORKER_REPO_ROOT": "/worker/repo with spaces",
        "ENVIRONMENT_ID": "",
        "SNAPSHOT_PATH": "",
        "WANDB_ENTITY": "",
        "DRY_RUN": "0",
    }

    def run(*selectors, **updates):
        result = subprocess.run(
            ["bash", str(SCRIPT), *selectors],
            env={**env, **updates},
            capture_output=True,
            text=True,
            timeout=10,
            cwd=tmp_path,
        )
        submitted = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, submitted

    return run


def test_all_experiments(scheduler):
    result, calls = scheduler()
    assert result.returncode == 0, result.stderr
    assert len(calls) == 10
    assert calls[-1] == ["-l"]
    expected = [
        [], ["loss.value_weight=0.5"], ["loss.consistency_weight=5"],
        ["replay.priority_alpha=0.6"],
        ["loss.value_weight=0.5", "loss.consistency_weight=5"],
        ["training.precision=fp32"], ["replay.per_mode=v2"],
        ["replay.final_per_mode=v2"],
        ["replay.priority_alpha=1", "replay.priority_beta_initial=0.26", "replay.priority_beta_final=0.65"],
    ]
    for call, overrides in zip(calls[:-1], expected, strict=True):
        assert call[:2] == ["bash", "-c"]
        assert call[4] == "/worker/repo with spaces"
        assert "environment.id=ALE/UpNDown-v5" in call
        assert "checkpoint.pre_final_snapshot_path=null" in call
        for override in overrides:
            assert override in call
    text = result.stdout + result.stderr + json.dumps(calls)
    assert "controller-secret" not in text
    assert "wandb-secret" not in text


@pytest.mark.parametrize("selectors", [("bad",), ("baseline", "bad"), ("all", "0")])
def test_invalid_selection_no_submission(scheduler, selectors):
    result, calls = scheduler(*selectors)
    assert result.returncode != 0
    assert not calls


def test_dedup_and_options(scheduler):
    result, calls = scheduler(
        "0", "baseline", "8", ENVIRONMENT_ID="Pong-v5",
        WANDB_ENTITY="team", SNAPSHOT_PATH="/worker/snapshot.pt",
    )
    assert result.returncode == 0
    assert len(calls) == 3
    assert "environment.id=ALE/Pong-v5" in calls[0]
    assert "wandb.entity=team" in calls[0]
    assert "checkpoint.resume_pre_final_path=/worker/snapshot.pt" in calls[0]


def test_requires_controller_key(scheduler):
    result, calls = scheduler("0", JOBD_API_KEY="")
    assert result.returncode != 0
    assert "JOBD_API_KEY" in result.stderr
    assert not calls


def test_dry_run_no_side_effects(scheduler, tmp_path):
    output = tmp_path / "not-created"
    result, calls = scheduler("0", DRY_RUN="1", JOBD_API_KEY="", RUN_ROOT=str(output))
    assert result.returncode == 0
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_command(scheduler, tmp_path, exit_code):
    repo = tmp_path / "worker repo"
    repo.mkdir()
    result, calls = scheduler("0", WORKER_REPO_ROOT=str(repo))
    assert result.returncode == 0
    binaries = tmp_path / "worker-bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text(f"#!/bin/sh\necho training-output\necho training-error >&2\nexit {exit_code}\n")
    uv.chmod(0o755)
    executed = subprocess.run(
        calls[0], env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        capture_output=True, text=True, timeout=10,
    )
    assert executed.returncode == exit_code
    log = (repo / "runs/test/baseline/training.log").read_text()
    assert "training-output" in log
    assert "training-error" in log
