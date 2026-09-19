"""Ensure a single pending self-stop using jobd's local Unix-socket API.

Runs on the worker. Never starts/restarts a daemon or opens its database.
All callers should use this helper; flock serializes cooperating submissions.
"""

import argparse
import fcntl
import http.client
import json
import os
import socket
from pathlib import Path


class LocalClient(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=20)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)

    def call(self, method, path, body=None):
        self.request(
            method,
            path,
            body=json.dumps(body) if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        response = self.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(
                "Local jobd request failed; no automatic submission retry"
            )
        return json.loads(raw)


def ensure_stop(call, instance_id):
    if not str(instance_id).isdigit():
        raise ValueError("Invalid instance ID")
    command = ["/root/.local/bin/vastai", "stop", "instance", str(instance_id)]
    pending = []
    offset = 0
    while True:
        page = call("GET", f"/jobs?limit=100&offset={offset}")
        jobs = page["jobs"]
        if not isinstance(jobs, list):
            raise TypeError("Invalid local queue response")
        for job in jobs:
            # Fail closed on malformed records, rather than assuming an empty queue.
            if not isinstance(job["command"], list) or not isinstance(
                job["status"], str
            ):
                raise TypeError("Invalid local job")
            argv = job["command"]
            matches = (
                len(argv) == 4
                and Path(argv[0]).name == "vastai"
                and argv[1:] == command[1:]
            )
            if matches and job["status"] in {"queued", "running"}:
                pending.append(job["id"])
        if len(jobs) < 100:
            break
        offset += len(jobs)
    if pending:
        return {"changed": False, "job_ids": pending}
    # Do not retry POST on failure: the daemon may have accepted it.
    job = call("POST", "/jobs", {"command": command})
    return {"changed": True, "job_ids": [job["id"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id")
    args = parser.parse_args()
    os.umask(0o077)
    state = Path(
        os.environ.get("JOBD_STATE_DIR", "~/.local/state/jobd-worker")
    ).expanduser()
    # Do not mkdir: missing worker state is an error.
    with (state / "ensure-stop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        client = LocalClient(state / "local/control.sock")
        try:
            print(json.dumps(ensure_stop(client.call, args.instance_id)))
        finally:
            client.close()


if __name__ == "__main__":
    main()
