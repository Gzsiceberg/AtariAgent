"""Test scheduler credential handling without W&B access or training."""

import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/schedule_tsp_experiments.sh"


@pytest.mark.parametrize(
    "number,name,override",
    [("5", "fp32", "training.precision=fp32"), ("6", "per_v2", "replay.per_mode=v2")],
)
@pytest.mark.parametrize("selector", ["number", "name", "all"])
def test_experiment(tmp_path, number, name, override, selector):
    selection = {"number": number, "name": name, "all": "all"}[selector]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    tsp = binaries / "tsp"
    tsp.write_text("#!/usr/bin/env bash\nexit 99\n")
    tsp.chmod(0o755)
    run_root = tmp_path / "runs"
    result = subprocess.run(
        ["bash", str(SCRIPT), selection],
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "WANDB_API_KEY": "test-secret",
            "RUN_ROOT": str(run_root),
            "RUN_ID": "test",
            "DRY_RUN": "1",
            "RUNPOD_POD_ID": "",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert override in (run_root / name / "job.sh").read_text()
    jobs = list(run_root.glob("*/job.sh"))
    assert len(jobs) == (7 if selection == "all" else 1)
    for job in jobs:
        if job.parent.name != name:
            assert override.split("=")[0] + "=" not in job.read_text()


@pytest.mark.parametrize("saved_login", [False, True])
def test_real_wandb_credential_lookup(tmp_path, saved_login):
    pytest.importorskip("wandb")
    home = tmp_path / "home"
    home.mkdir()
    netrc = home / ".netrc"
    secret = "a" * 40
    if saved_login:
        netrc.write_text(f"machine api.wandb.ai login user password {secret}\n")
        netrc.chmod(0o600)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("WANDB_")
    }
    env.update(
        HOME=str(home),
        NETRC=str(netrc),
        WANDB_CONFIG_DIR=str(home / "config"),
        WANDB_BASE_URL="https://api.wandb.ai",
    )
    # Exercise the real CLI with stdin disabled, as in the scheduler.
    result = subprocess.run(
        ["wandb", "login", "--no-verify"],
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=20,
    )
    assert (result.returncode == 0) == saved_login, result.stderr
    assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "key,status,user_input,success,prompt,lookup",
    [
        ("existing-secret", 99, "", True, False, False),
        ("", 0, "", True, False, True),
        ("", 10, "entered-secret\n", True, True, True),
        ("", 10, "\nentered-secret\n", True, True, True),
        ("", 10, "", False, True, True),
        ("", 1, "", False, True, True),
    ],
)
def test_wandb_credentials(tmp_path, key, status, user_input, success, prompt, lookup):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {
        "tsp": "exit 99\n",
        "uv": (
            '[[ "$*" == "run wandb login --no-verify" ]] || exit 99\n'
            'touch "$AUTH_CHECK_MARKER"\nexit "$AUTH_STATUS"\n'
        ),
    }.items():
        path = binaries / name
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)
    marker = tmp_path / "auth-checked"
    run_root = tmp_path / "runs"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "WANDB_API_KEY": key,
        "AUTH_STATUS": str(status),
        "AUTH_CHECK_MARKER": str(marker),
        "RUN_ROOT": str(run_root),
        "RUN_ID": "test",
        "DRY_RUN": "1",
        "RUNPOD_POD_ID": "",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "3"],
        env=env,
        input=user_input,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == success
    assert ("W&B API key: " in result.stderr) == prompt
    assert marker.exists() == lookup
    job = run_root / "priority_alpha/job.sh"
    assert job.exists() == success
    output = result.stdout + result.stderr + (job.read_text() if job.exists() else "")
    assert "existing-secret" not in output
    assert "entered-secret" not in output
    if success:
        assert "replay.priority_alpha=0.6" in job.read_text()
