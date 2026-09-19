"""Verify worker-supplied W&B credentials without creating a run or retaining login files."""

import os
import socket
import tempfile
from pathlib import Path


def main() -> None:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("WANDB_API_KEY was not supplied by jobd-worker")

    # W&B login can write netrc/config files. Isolate them and remove on exit.
    with tempfile.TemporaryDirectory(prefix="atari-wandb-auth-") as directory:
        overrides = {
            "HOME": directory,
            "NETRC": str(Path(directory) / ".netrc"),
            "WANDB_CONFIG_DIR": str(Path(directory) / "config"),
            "WANDB_CACHE_DIR": str(Path(directory) / "cache"),
            "WANDB_MODE": "online",
        }
        previous = {name: os.environ.get(name) for name in overrides}
        try:
            os.environ.update(overrides)
            import wandb

            if not wandb.login(key=key, verify=True, timeout=30):
                raise RuntimeError("W&B authentication failed")
            print(f"W&B authentication verified on {socket.gethostname()}")
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    main()
