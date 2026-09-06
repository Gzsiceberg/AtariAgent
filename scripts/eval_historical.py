#!/usr/bin/env python3
"""Screen historical Asterix checkpoints using a common standalone protocol.

Run from any directory with uv run python /path/to/scripts/eval_historical.py.
Evaluations run sequentially in separate processes. No training is performed.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def command_for(checkpoint: Path, output: Path, cfg: DictConfig) -> list[str]:
    return [
        sys.executable, "-u", str(REPO / "scripts/eval_agent.py"), str(checkpoint),
        "--episodes", str(cfg.episodes), "--num-envs", str(cfg.num_envs),
        "--seed", str(cfg.seed), "--device", str(cfg.device),
        "--search-algorithm", "puct", "--num-simulations", str(cfg.num_simulations),
        "--output-json", str(output / "results.json"),
    ]


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def summarize(result: dict, episodes: int) -> dict:
    rewards = np.asarray(result["rewards"], dtype=np.float64)
    if rewards.shape != (episodes,) or not np.isfinite(rewards).all():
        raise ValueError("expected exactly the requested number of finite returns")
    rng = np.random.default_rng(42)
    # Bounded memory even if the evaluation episode count is increased.
    means = np.concatenate([
        rng.choice(rewards, size=(100, episodes)).mean(axis=1)
        for _ in range(100)
    ])
    return {
        "mean": float(rewards.mean()), "median": float(np.median(rewards)),
        "std": float(rewards.std()), "min": float(rewards.min()),
        "max": float(rewards.max()),
        "mean_episode_bootstrap_95_ci": np.quantile(means, [.025, .975]).tolist(),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="eval_historical")
def main(cfg: DictConfig) -> None:
    for key in ("episodes", "num_envs", "num_simulations"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not cfg.checkpoints:
        raise ValueError("at least one checkpoint is required")
    output = resolve_path(str(cfg.output_dir))
    checkpoints = {}
    for name, value in cfg.checkpoints.items():
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"unsafe checkpoint name: {name!r}")
        path = resolve_path(str(value))
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[name] = path
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}; set output_dir to a new path")
    if cfg.dry_run:
        for name, checkpoint in checkpoints.items():
            print(shlex.join(command_for(checkpoint, output / name, cfg)))
        return

    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (output / "source.diff").write_text(subprocess.check_output(
        ["git", "diff", "HEAD"], cwd=REPO, text=True,
    ))
    # Preserve this launcher even if it has not been committed yet.
    (output / "launcher.py").write_text(Path(__file__).read_text())
    summary = {
        "protocol": OmegaConf.to_container(cfg, resolve=True),
        "evaluation_git_commit": commit,
        "note": "Screening only. Episode bootstrap intervals do not include training-seed "
                "uncertainty or checkpoint-selection bias. Confirm the selected model on fresh starts.",
        "bootstrap_resamples": 10000, "bootstrap_seed": 42,
        "results": {}, "failures": [],
    }
    for name, checkpoint in checkpoints.items():
        folder = output / name
        folder.mkdir()
        command = command_for(checkpoint, folder, cfg)
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
            "command": command, "evaluation_git_commit": commit,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json(folder / "metadata.json", metadata)
        print(f"\nEvaluating {name}: {folder}", flush=True)
        with (folder / "evaluation.log").open("w") as log:
            with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
        metadata.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
        save_json(folder / "metadata.json", metadata)
        try:
            if code:
                raise RuntimeError(f"evaluator exited with code {code}")
            result = json.loads((folder / "results.json").read_text())
            stats = summarize(result, cfg.episodes)
            with (folder / "episodes.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["episode", "seed", "reward"])
                for index, reward in enumerate(result["rewards"]):
                    writer.writerow([index + 1, cfg.seed + index, reward])
            summary["results"][name] = stats
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            summary["failures"].append({"name": name, "error": str(error)})
        save_json(output / "summary.json", summary)

    print("\nCheckpoint                         Mean       Median      95% episode CI")
    for name, stats in summary["results"].items():
        low, high = stats["mean_episode_bootstrap_95_ci"]
        print(f"{name:32} {stats['mean']:10.1f} {stats['median']:10.1f}  [{low:.1f}, {high:.1f}]")
    print(f"\nResults: {output}")
    if summary["failures"]:
        raise SystemExit("Some evaluations failed; inspect summary.json and evaluation.log")


if __name__ == "__main__":
    main()
