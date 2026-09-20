"""Test jobd submissions without a controller or actual training."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_game_experiments.sh"


@pytest.fixture
def batch(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    jobd = binaries / "jobd"
    jobd.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(int(os.environ.get('SUBMIT_FAILURE', '0')))\n"
    )
    jobd.chmod(0o755)
    uv = binaries / "uv"
    uv.write_text('#!/bin/bash\nprintf "training output\\n"\nexit "${TRAIN_FAILURE:-0}"\n')
    uv.chmod(0o755)
    repo = tmp_path / "worker checkout"
    repo.mkdir()
    return {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "CALLS": str(tmp_path / "calls.jsonl"),
        "WORKER_REPO_ROOT": str(repo),
        "RUN_ROOT": "runs with spaces/test",
        "RUN_ID": "test-run",
        "START_GAME": "1",
        "END_GAME": "26",
        "SEED": "2",
        "DRY_RUN": "0",
        "WANDB_PROJECT": "test-project",
        "WANDB_ENTITY": "test-entity",
    }


def run_batch(env, *options):
    return subprocess.run(
        ["bash", str(SCRIPT), *options], env=env, capture_output=True, text=True, timeout=15,
    )


def calls(env):
    return [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]


def test_all_games_are_queued_without_local_writes(batch):
    result = run_batch(batch)
    assert result.returncode == 0, result.stderr
    jobs = calls(batch)
    assert jobs[-1] == ["-l"]
    assert len(jobs) == 27
    assert len({job[5] for job in jobs[:-1]}) == 26
    assert "environment.id=ALE/Asterix-v5" in jobs[0]
    assert "environment.id=ALE/UpNDown-v5" in jobs[-2]
    assert not (Path(batch["WORKER_REPO_ROOT"]) / batch["RUN_ROOT"]).exists()
    for job in jobs[:-1]:
        assert job[:2] == ["bash", "-ec"]
        assert "uv run --no-sync python scripts/train_agent.py" in job[2]
        assert "seed=2" in job
        assert "checkpoint.pre_final_snapshot_path=null" in job
        assert "wandb.entity=test-entity" in job
        assert "wandb.project=test-project" in job
        assert not any(arg.startswith("search=") for arg in job)
        assert not any(arg.startswith("reanalysis.target_update_interval=") for arg in job)
        assert job[5].endswith("/default")


def test_dry_run_has_no_side_effects(batch):
    result = run_batch({**batch, "DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Command:") == 26
    assert "nothing submitted or written" in result.stdout
    assert not Path(batch["CALLS"]).exists()
    assert not (Path(batch["WORKER_REPO_ROOT"]) / batch["RUN_ROOT"]).exists()


def test_ranges_cover_all_games_with_isolated_outputs(batch):
    batch.pop("RUN_ROOT")
    for start, end in ((1, 9), (10, 18), (19, 26)):
        env = {**batch, "START_GAME": str(start), "END_GAME": str(end)}
        result = run_batch(env)
        assert result.returncode == 0, result.stderr
        jobs = calls(env)[-(end - start + 2):-1]
        assert len(jobs) == end - start + 1
        assert all(f"/games_{start}-{end}/" in job[5] for job in jobs)
    jobs = [job for job in calls(batch) if job != ["-l"]]
    assert len({job[5] for job in jobs}) == 26


@pytest.mark.parametrize("start,end", [
    ("0", "9"), ("10", "9"), ("1", "27"), ("abc", "26"),
    ("01", "9"), ("1", "999999999999999999999"),
])
def test_invalid_game_range(batch, start, end):
    result = run_batch({**batch, "START_GAME": start, "END_GAME": end})
    assert result.returncode != 0
    assert "Game range must satisfy" in result.stderr
    assert not Path(batch["CALLS"]).exists()


@pytest.mark.parametrize("seed", ["-1", "abc", "4294967296", "99999999999999999999"])
def test_invalid_seed(batch, seed):
    result = run_batch({**batch, "SEED": seed})
    assert result.returncode != 0
    assert "SEED must be" in result.stderr
    assert not Path(batch["CALLS"]).exists()


@pytest.mark.parametrize("failure", [0, 7])
def test_worker_logging_exit_status_and_overwrite_protection(batch, failure):
    env = {**batch, "START_GAME": "26", "END_GAME": "26", "SEED": "3"}
    assert run_batch(env).returncode == 0
    job, listing = calls(env)
    assert listing == ["-l"]
    assert "seed=3" in job
    assert "wandb.name=UpNDown-v5_default_seed3_test-run" in job
    result = subprocess.run(job, env={**env, "TRAIN_FAILURE": str(failure)}, timeout=15)
    assert result.returncode == failure
    output = Path(env["WORKER_REPO_ROOT"]) / env["RUN_ROOT"] / "up-n-down/default"
    assert (output / "training.log").read_text() == "training output\n"
    result = subprocess.run(job, env=env, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert (output / "training.log").read_text() == "training output\n"


def test_gumbel_search(batch):
    result = run_batch(batch, "--gumbel")
    assert result.returncode == 0, result.stderr
    jobs = calls(batch)[:-1]
    assert len(jobs) == 26
    for job in jobs:
        assert "search=gumbel" in job
        assert job[5].endswith("/gumbel")
        assert f"checkpoint.path={job[5]}/checkpoints/agent_latest.pt" in job
        assert any(arg.startswith("wandb.name=") and "_gumbel_seed2_" in arg for arg in job)
        assert any(arg.startswith("wandb.tags=") and ",gumbel," in arg for arg in job)


@pytest.mark.parametrize("option,code", [("--help", 0), ("--unknown", 1)])
def test_options_do_not_submit_jobs(batch, option, code):
    result = run_batch(batch, option)
    assert result.returncode == code
    assert not Path(batch["CALLS"]).exists()


def test_gumbel_dry_run(batch):
    result = run_batch({**batch, "DRY_RUN": "1"}, "--gumbel")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("search=gumbel") == 26
    assert not Path(batch["CALLS"]).exists()


@pytest.mark.parametrize("search", ["puct", "gumbel"])
def test_target400_experiment(batch, search):
    result = run_batch(batch, f"--{search}-target-400")
    assert result.returncode == 0, result.stderr
    jobs = calls(batch)[:-1]
    assert len(jobs) == 26
    experiment = f"{search}_target400"
    for job in jobs:
        assert f"search={search}" in job
        assert "reanalysis.target_update_interval=400" in job
        assert job[5].endswith(f"/{experiment}")
        assert f"checkpoint.path={job[5]}/checkpoints/agent_latest.pt" in job
        assert f"evaluation.data_path={job[5]}/evaluations/agent_evaluations.json" in job
        assert any(arg.startswith("wandb.name=") and f"_{experiment}_seed2_" in arg for arg in job)
        assert any(arg.startswith("wandb.tags=") and f",{experiment}," in arg for arg in job)


@pytest.mark.parametrize("options", [
    ("--gumbel", "--puct-target-400"),
    ("--gumbel-target-400", "--gumbel"),
    ("--puct-target-400", "--gumbel-target-400"),
    ("--puct-target-400", "--puct-target-400"),
])
def test_conflicting_experiment_options(batch, options):
    result = run_batch(batch, *options)
    assert result.returncode != 0
    assert "choose only one experiment" in result.stderr
    assert not Path(batch["CALLS"]).exists()


@pytest.mark.parametrize("search", ["puct", "gumbel"])
def test_target400_dry_run(batch, search):
    result = run_batch({**batch, "DRY_RUN": "1"}, f"--{search}-target-400")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("reanalysis.target_update_interval=400") == 26
    assert result.stdout.count(f"search={search}") == 26
    assert not Path(batch["CALLS"]).exists()


def test_submission_failure_is_not_retried(batch):
    result = run_batch({**batch, "SUBMIT_FAILURE": "9"})
    assert result.returncode == 9
    assert len(calls(batch)) == 1
