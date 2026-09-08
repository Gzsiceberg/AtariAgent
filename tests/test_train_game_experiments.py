"""Exercise the scheduler with isolated tsp queues and no actual training."""

import csv
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("tsp"), reason="Task Spooler not installed")


@pytest.fixture
def batch(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in (
        "train_game_experiments.sh", "atari_100k_paper_scores.csv",
        "summarize_game_experiments.py",
    ):
        shutil.copy2(SOURCE_ROOT / "scripts" / name, scripts / name)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    mocks = {
        **{
            name: '#!/usr/bin/env bash\nprintf "%s\\n" "$0" >> "$UNEXPECTED_CALLS"\nexit 99\n'
            for name in ("sudo", "systemd-inhibit", "systemctl", "nvidia-smi")
        },
        "uv": '''#!/usr/bin/env bash
if [[ "$1 $2 $3" == 'run python scripts/summarize_game_experiments.py' ]]; then
    shift 2
    exec "$TEST_PYTHON" "$@"
fi
[[ "$1 $2 $3" == 'run python scripts/train_agent.py' ]] || {
    printf '%s\\n' "$*" >> "$UNEXPECTED_CALLS"
    exit 99
}
for arg in "$@"; do
    if [[ "$arg" == environment.id=* ]]; then
        printf '%s\\n' "$arg" >> "$TRAINING_CALLS"
        printf 'training stdout\\n'
        printf 'training stderr\\n' >&2
        if [[ "$arg" == "environment.id=${FAIL_GAME:-}" ]]; then exit 7; fi
        if [[ "$arg" == "environment.id=${BLOCK_GAME:-}" ]]; then
            while [[ ! -e "$RELEASE_TRAINING" ]]; do sleep 0.05; done
        fi
    fi
done
''',
    }
    for name, content in mocks.items():
        path = binaries / name
        path.write_text(content)
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "RUN_ID": "test-run",
        "SEED": "2",
        "TEST_PYTHON": sys.executable,
        "RUN_ROOT": str(tmp_path / "runs with spaces"),
        "DRY_RUN": "0",
        "WANDB_PROJECT": "test-project",
        "WANDB_ENTITY": "test-entity",
        "TRAINING_CALLS": str(tmp_path / "training_calls"),
        "UNEXPECTED_CALLS": str(tmp_path / "unexpected_calls"),
        "TS_SOCKET": str(tmp_path / "tsp.socket"),
        "TS_SLOTS": "1",
        "FAIL_GAME": "",
        "BLOCK_GAME": "",
        "RELEASE_TRAINING": str(tmp_path / "release_training"),
    }
    yield tmp_path, env
    (tmp_path / "release").touch()
    Path(env["RELEASE_TRAINING"]).touch()
    if Path(env["TS_SOCKET"]).exists():
        subprocess.run(["tsp", "-S", "1"], env=env, check=True, timeout=10)
        jobs_path = Path(env["RUN_ROOT"]) / "tsp_jobs.tsv"
        if jobs_path.exists():
            for line in jobs_path.read_text().splitlines():
                subprocess.run(
                    ["tsp", "-w", line.split("\t")[0]], env=env,
                    check=False, timeout=10,
                )
        subprocess.run(["tsp", "-K"], env=env, check=False, timeout=10)
    assert not Path(env["UNEXPECTED_CALLS"]).exists()


def run_batch(root, env):
    return subprocess.run(
        ["bash", str(root / "scripts/train_game_experiments.sh")],
        env=env, capture_output=True, text=True, timeout=15,
    )


@pytest.mark.parametrize("run_id, suffix", [(None, ""), ("seed_2", ""), ("test-run", "_test-run")])
def test_wandb_name_does_not_duplicate_default_seed(batch, run_id, suffix):
    root, env = batch
    env = {**env, "DRY_RUN": "1"}
    if run_id is None:
        env.pop("RUN_ID")
    else:
        env["RUN_ID"] = run_id
    result = run_batch(root, env)
    assert result.returncode == 0, result.stderr
    worker = Path(env["RUN_ROOT"]) / "battle-zone/default/job.sh"
    args = shlex.split(worker.read_text().splitlines()[3])
    assert (
        "wandb.name='${environment_slug:${environment.id}}_default_seed${seed}"
        f"{suffix}'"
    ) in args


def test_dry_run_generates_simple_jobs(batch):
    root, env = batch
    result = run_batch(root, {**env, "DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Command: ") == 26
    assert not Path(env["TS_SOCKET"]).exists()
    assert not Path(env["TRAINING_CALLS"]).exists()
    workers = list(Path(env["RUN_ROOT"]).glob("*/default/job.sh"))
    assert len(workers) == 26
    for worker in workers:
        subprocess.run(["bash", "-n", str(worker)], check=True)
        content = worker.read_text()
        assert len(content.splitlines()) == 4
        assert "seed=2" in content
        assert "checkpoint.pre_final_snapshot_path=null" in content
        assert "wandb.entity=test-entity" in content
        assert "wandb.project=test-project" in content
        assert "2>&1 | tee " in content


@pytest.mark.parametrize("fail", [False, True])
def test_games_run_once_and_failure_does_not_block_queue(batch, fail):
    root, env = batch
    if fail:
        env["FAIL_GAME"] = "ALE/BankHeist-v5"
    result = run_batch(root, env)
    assert result.returncode == 0, result.stderr
    run_root = Path(env["RUN_ROOT"])
    jobs = (run_root / "tsp_jobs.tsv").read_text().splitlines()
    assert len(jobs) == 27
    assert jobs[-1].endswith("\tsummary")
    assert len({line.split("\t")[0] for line in jobs}) == 27
    outcomes = [
        subprocess.run(
            ["tsp", "-w", line.split("\t")[0]], env=env, timeout=15,
            check=False,
        ).returncode
        for line in jobs
    ]
    expected_outcomes = [0] * 27
    if fail:
        expected_outcomes[1] = 7
    assert outcomes == expected_outcomes
    expected_calls = [
        f"environment.id={line.split('|')[1]}"
        for line in (run_root / "games.tsv").read_text().splitlines()
    ]
    assert Path(env["TRAINING_CALLS"]).read_text().splitlines() == expected_calls
    for log in run_root.glob("*/default/training.log"):
        assert log.read_text() == "training stdout\ntraining stderr\n"
    assert len(list(run_root.glob("*/default/training.log"))) == 26
    assert not (run_root / "batch_status.txt").exists()
    with (run_root / "results.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 28
    assert {"human_paper", "efficientzero_paper", "efficientzero_paper_v2"} <= rows[0].keys()


def test_launcher_exits_without_waiting_or_changing_slots(batch):
    root, env = batch
    subprocess.run(["tsp", "-S", "3"], env=env, check=True, timeout=10)
    subprocess.run(
        ["tsp", "-N", "3", "bash", "-c",
         'while [[ ! -e "$1" ]]; do sleep 0.05; done', "bash", str(root / "release")],
        env=env, check=True, timeout=10,
    )
    result = run_batch(root, env)
    assert result.returncode == 0, result.stderr
    assert subprocess.check_output(["tsp", "-S"], env=env, text=True).strip() == "3"
    assert not Path(env["TRAINING_CALLS"]).exists()
    jobs = (Path(env["RUN_ROOT"]) / "tsp_jobs.tsv").read_text().splitlines()
    assert len(jobs) == 27
    for line in jobs:
        assert subprocess.check_output(
            ["tsp", "-s", line.split("\t")[0]], env=env, text=True,
        ).strip() == "queued"


def test_rerun_skips_completed_games_and_schedules_missing_or_incomplete(batch):
    root, env = batch
    assert run_batch(root, {**env, "DRY_RUN": "1"}).returncode == 0
    run_root = Path(env["RUN_ROOT"])
    games = (run_root / "games.tsv").read_text().splitlines()
    for line in games:
        game = line.split("|")[0]
        if game == "alien":
            continue  # Missing evaluation.
        update = 119999 if game == "bank-heist" else 120000
        path = run_root / game / "default/evaluations/agent_evaluations.json"
        path.write_text(json.dumps({"evaluations": [
            {"update": update, "mean": 100, "median": 90, "std": 10},
        ]}))
    result = run_batch(root, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Skipping completed game:") == 24
    jobs = (run_root / "tsp_jobs.tsv").read_text().splitlines()
    assert [line.split("\t")[1] for line in jobs] == ["bank-heist", "alien", "summary"]
    for line in jobs:
        subprocess.run(["tsp", "-w", line.split("\t")[0]], env=env, check=True, timeout=10)
    with (run_root / "results.csv").open() as source:
        rows = {row["game"]: row for row in csv.DictReader(source)}
    assert rows["alien"]["status"] == "pending"
    assert rows["bank-heist"]["status"] == "partial"
    assert rows["asterix"]["atariagent_mean"] == "100.0"
    with (run_root / "paper_scores.csv").open() as source:
        reader = csv.DictReader(source)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        reference = next(row for row in reader if row["game"].strip() == "asterix")
    assert float(rows["asterix"]["efficientzero_paper_v2"]) == float(reference["efficientzero_v2"])
    # Mark remaining games complete; rerun must enqueue only a fresh summary.
    for game in ("bank-heist", "alien"):
        path = run_root / game / "default/evaluations/agent_evaluations.json"
        path.write_text(json.dumps({"evaluations": [
            {"update": 120000, "mean": 200, "median": 190, "std": 10},
        ]}))
    result = run_batch(root, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Skipping completed game:") == 26
    updated_jobs = (run_root / "tsp_jobs.tsv").read_text().splitlines()
    assert updated_jobs[:-1] == jobs
    assert updated_jobs[-1].endswith("\tsummary")
    subprocess.run(
        ["tsp", "-w", updated_jobs[-1].split("\t")[0]], env=env, check=True, timeout=10,
    )
    with (run_root / "results.csv").open() as source:
        rows = {row["game"]: row for row in csv.DictReader(source)}
    assert rows["alien"]["status"] == "complete"
    assert rows["alien"]["atariagent_mean"] == "200.0"
    assert rows["normed-mean"]["aggregate_game_count"] == "26"
    assert not (run_root / "summary-job.sh").exists()
    assert not list(run_root.glob("summary-job.*.sh"))
    assert not list(run_root.glob(".*.lock"))


def test_summary_is_queued_directly_after_games(batch):
    root, env = batch
    env["BLOCK_GAME"] = "ALE/Asterix-v5"
    result = run_batch(root, env)
    assert result.returncode == 0, result.stderr
    run_root = Path(env["RUN_ROOT"])
    jobs = (run_root / "tsp_jobs.tsv").read_text().splitlines()
    summary_id = jobs[-1].split("\t")[0]
    assert subprocess.check_output(
        ["tsp", "-s", summary_id], env=env, text=True,
    ).strip() == "queued"
    info = subprocess.check_output(["tsp", "-i", summary_id], env=env, text=True)
    assert "Command: uv run python scripts/summarize_game_experiments.py summarize" in info
    assert not (run_root / "results.csv").exists()
    Path(env["RELEASE_TRAINING"]).touch()
    subprocess.run(["tsp", "-w", summary_id], env=env, check=True, timeout=10)
    assert (run_root / "results.csv").exists()


def test_seed_defaults_and_explicit_override_are_isolated(batch):
    root, env = batch
    env = {**env, "DRY_RUN": "1"}
    env.pop("RUN_ID")
    env.pop("RUN_ROOT")
    for seed in (2, 3):
        result = run_batch(root, {**env, "SEED": str(seed)})
        assert result.returncode == 0, result.stderr
        run_root = root / f"runs/game_experiments/seed_{seed}"
        assert (run_root / "seed.txt").read_text().strip() == str(seed)
        for worker in run_root.glob("*/default/job.sh"):
            assert f"seed={seed}" in worker.read_text()
    # Reusing an explicit root with a different seed must not skip/overwrite it.
    result = run_batch(root, {
        **env, "SEED": "3",
        "RUN_ROOT": str(root / "runs/game_experiments/seed_2"),
    })
    assert result.returncode != 0
    assert "belongs to seed 2" in result.stderr


@pytest.mark.parametrize("seed", ["-1", "abc", "4294967296", "99999999999999999999"])
def test_invalid_seed(batch, seed):
    root, env = batch
    result = run_batch(root, {**env, "SEED": seed})
    assert result.returncode != 0
    assert "SEED must be" in result.stderr
