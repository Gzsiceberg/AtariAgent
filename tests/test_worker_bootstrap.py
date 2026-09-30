"""Offline tests: never install packages, contact W&B, or start real workers."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bootstrap_env(tmp_path):
    binaries = tmp_path / ".local/bin"
    binaries.mkdir(parents=True)
    scripts = {
        "pgrep": 'exit "${WORKER_RUNNING:-1}"',
        "uv": 'echo "uv $*" >> "$TRACE"; if [ "$1" = run ]; then exit "${READINESS_EXIT:-0}"; fi; exit "${UV_EXIT:-0}"',
        "curl": 'echo install >> "$TRACE"; exit "${CURL_EXIT:-0}"',
        "jobd": 'echo "jobd $*" >> "$TRACE"; if [ "$1" = auth ]; then exit "${VERIFY_EXIT:-0}"; fi',
    }
    for name, script in scripts.items():
        path = binaries / name
        path.write_text("#!/bin/bash\n" + script + "\n")
        path.chmod(0o700)
    env = {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "TRACE": str(tmp_path / "trace"),
        "JOBD_WORKER_TOKEN": "test-secret",
    }
    return env


def run_bootstrap(env):
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "bootstrap.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert "test-secret" not in result.stdout + result.stderr
    trace_path = Path(env["TRACE"])
    trace = trace_path.read_text() if trace_path.exists() else ""
    return result, trace


def test_bootstrap_prepares_without_starting_worker(bootstrap_env):
    result, trace = run_bootstrap(bootstrap_env)
    assert result.returncode == 0
    assert trace.splitlines() == [
        "uv sync --frozen --extra wandb",
        "uv run --no-sync python -",
        "install",
        "jobd auth verify-worker-token",
    ]
    assert "Preparation complete" in result.stdout
    marker = Path(bootstrap_env["HOME"]) / ".local/state/atariagent/prepared-revision"
    assert len(marker.read_text().strip()) == 40
    assert "jobd worker" not in trace


@pytest.mark.parametrize(
    "failure", ["UV_EXIT", "READINESS_EXIT", "CURL_EXIT", "VERIFY_EXIT"]
)
def test_failure_prevents_worker_start(bootstrap_env, failure):
    bootstrap_env[failure] = "1"
    result, trace = run_bootstrap(bootstrap_env)
    assert result.returncode != 0
    assert "jobd worker" not in trace
    assert "Bootstrap failed" in result.stderr
    assert not (
        Path(bootstrap_env["HOME"]) / ".local/state/atariagent/prepared-revision"
    ).exists()


def test_existing_worker_is_not_interrupted(bootstrap_env):
    bootstrap_env["WORKER_RUNNING"] = "0"
    result, trace = run_bootstrap(bootstrap_env)
    assert result.returncode != 0
    assert not trace


def test_missing_token_fails_before_installation(bootstrap_env):
    del bootstrap_env["JOBD_WORKER_TOKEN"]
    result, trace = run_bootstrap(bootstrap_env)
    assert result.returncode != 0
    assert not trace
