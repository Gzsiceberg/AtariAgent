"""Offline tests: never install packages, contact W&B, or start real workers."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_bootstrap_verifies_before_start(bootstrap_env):
    result, trace = run_bootstrap(bootstrap_env)
    assert result.returncode == 0
    assert trace.splitlines() == [
        "uv sync --frozen --extra wandb",
        "uv run --no-sync python -",
        "install",
        "jobd auth verify-worker-token",
        "jobd worker start",
    ]
    assert "Bootstrap complete" in result.stdout


@pytest.mark.parametrize("failure", ["UV_EXIT", "READINESS_EXIT", "CURL_EXIT", "VERIFY_EXIT"])
def test_failure_prevents_worker_start(bootstrap_env, failure):
    bootstrap_env[failure] = "1"
    result, trace = run_bootstrap(bootstrap_env)
    assert result.returncode != 0
    assert "jobd worker" not in trace
    assert "Bootstrap failed" in result.stderr


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


@pytest.mark.parametrize("login_success", [True, False])
def test_auth_check_isolates_login_files(monkeypatch, login_success):
    spec = importlib.util.spec_from_file_location(
        "verify_wandb_auth", ROOT / "scripts/verify_wandb_auth.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("WANDB_API_KEY", "test-secret")
    monkeypatch.setenv("HOME", "/original-home")
    homes = []

    def login(**kwargs):
        assert kwargs == {"key": "test-secret", "verify": True, "timeout": 30}
        homes.append(Path(os.environ["HOME"]))
        assert homes[-1] != Path("/original-home")
        Path(os.environ["NETRC"]).write_text("test-secret")
        return login_success

    # Deliberately provide no init() API: creating a run would fail the test.
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(login=login))
    if login_success:
        module.main()
    else:
        with pytest.raises(RuntimeError, match="authentication failed"):
            module.main()
    assert os.environ["HOME"] == "/original-home"
    assert not homes[0].exists()
