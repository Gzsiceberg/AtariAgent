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
    assert len(calls) == 4
    assert calls[-1] == ["-l"]
    expected = [
        ["search=gumbel", "reanalysis.target_update_interval=400", "reanalysis.cache_targets=false"],
        ["search=gumbel", "reanalysis.target_update_interval=200", "reanalysis.cache_targets=true"],
        ["search=gumbel", "reanalysis.target_update_interval=200", "reanalysis.cache_targets=true", "model.action_embedding=false"],
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


@pytest.mark.parametrize("selectors", [
    ("bad",), ("0", "bad"), ("all", "0"),
    *[(name,) for name in (
        "baseline", "value_loss_coeff", "per_v2", "mixed_value_threshold",
        "target_update_interval", "gumbel", "6", "9", "10", "11",
    )],
])
def test_invalid_selection_no_submission(scheduler, selectors):
    result, calls = scheduler(*selectors)
    assert result.returncode != 0
    assert not calls


def test_dedup_and_options(scheduler):
    result, calls = scheduler(
        "0", "gumbel_target400_nocache", "1", "gumbel_target200_cache", ENVIRONMENT_ID="Pong-v5",
        WANDB_ENTITY="team", SNAPSHOT_PATH="/worker/snapshot.pt",
    )
    assert result.returncode == 0
    assert len(calls) == 3
    assert "environment.id=ALE/Pong-v5" in calls[0]
    assert "wandb.entity=team" in calls[0]
    assert "checkpoint.resume_pre_final_path=/worker/snapshot.pt" in calls[0]


def test_v1_action_option_and_alias_are_deduplicated(scheduler):
    result, calls = scheduler("2", "gumbel_target200_cache_v1action")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert calls[-1] == ["-l"]
    for override in (
        "search=gumbel", "reanalysis.target_update_interval=200",
        "reanalysis.cache_targets=true", "model.action_embedding=false",
    ):
        assert override in calls[0]
    assert calls[0][5] == "runs/test/gumbel_target200_cache_v1action"
    assert any(arg.startswith("wandb.name=") and "gumbel_target200_cache_v1action" in arg
               for arg in calls[0])


def test_delegates_authentication_to_jobd(scheduler):
    result, calls = scheduler("0", JOBD_API_KEY="")
    assert result.returncode == 0
    assert len(calls) == 2


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
    log = (repo / "runs/test/gumbel_target400_nocache/training.log").read_text()
    assert "training-output" in log
    assert "training-error" in log
