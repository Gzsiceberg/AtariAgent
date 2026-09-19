"""Offline tests: never contact Vast or submit to a real jobd worker."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ensure_local_stop", ROOT / "scripts/ensure_local_stop.py"
)
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)


def job(status="queued", instance="123", job_id="local-1"):
    return {
        "id": job_id,
        "status": status,
        "command": ["/root/.local/bin/vastai", "stop", "instance", instance],
    }


@pytest.mark.parametrize(
    "status,submit",
    [("queued", False), ("running", False), ("succeeded", True), ("failed", True)],
)
def test_skip_only_pending_stop(status, submit):
    calls = []

    def call(method, path, body=None):
        calls.append((method, body))
        if method == "GET":
            return {"jobs": [job(status)]}
        assert body["command"][-1] == "123"
        return {"id": "local-2"}

    assert stop.ensure_stop(call, "123")["changed"] == submit
    assert sum(method == "POST" for method, _ in calls) == int(submit)


def test_all_pages_are_inspected():
    calls = []

    def call(method, path, body=None):
        assert method == "GET"
        calls.append(path)
        return {"jobs": [job("succeeded")] * 100 if "offset=0" in path else [job()]}

    assert stop.ensure_stop(call, "123") == {"changed": False, "job_ids": ["local-1"]}
    assert calls == ["/jobs?limit=100&offset=0", "/jobs?limit=100&offset=100"]


def test_stop_for_other_instance_does_not_suppress_submission():
    def call(method, path, body=None):
        if method == "GET":
            return {"jobs": [job(instance="456")]}
        return {"id": "local-2"}

    assert stop.ensure_stop(call, "123") == {"changed": True, "job_ids": ["local-2"]}


def test_existing_duplicates_are_reported_not_removed():
    def call(method, path, body=None):
        assert method == "GET"
        return {"jobs": [job(), job(job_id="local-2")]}

    assert stop.ensure_stop(call, "123")["job_ids"] == ["local-1", "local-2"]


def test_failed_inspection_never_submits():
    def call(method, path, body=None):
        assert method == "GET"
        raise ConnectionError("worker unavailable")

    with pytest.raises(ConnectionError):
        stop.ensure_stop(call, "123")


@pytest.mark.parametrize("payload", [{}, {"jobs": None}, {"jobs": [{}]}])
def test_malformed_inspection_never_submits(payload):
    def call(method, path, body=None):
        assert method == "GET"
        return payload

    with pytest.raises((KeyError, TypeError)):
        stop.ensure_stop(call, "123")


def test_uncertain_submission_is_not_retried():
    submissions = []

    def call(method, path, body=None):
        if method == "GET":
            return {"jobs": []}
        submissions.append(body)
        raise TimeoutError("lost response")

    with pytest.raises(TimeoutError):
        stop.ensure_stop(call, "123")
    assert len(submissions) == 1


def test_invalid_instance_id():
    with pytest.raises(ValueError):
        stop.ensure_stop(
            lambda *a: pytest.fail("invalid ID must not query"), "123;false"
        )
