#!/usr/bin/env python3
"""Summarize and inspect AtariAgent multi-game experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_evaluations(path: Path) -> list[dict[str, Any]]:
    """Load evaluation records from a training output file."""
    data = json.loads(path.read_text())
    evaluations = data.get("evaluations")
    if not isinstance(evaluations, list):
        raise TypeError(f"{path} does not contain an evaluations list")
    if not all(isinstance(evaluation, dict) for evaluation in evaluations):
        raise TypeError(f"{path} contains an invalid evaluation record")
    return evaluations


def is_complete(path: Path, expected_final_update: int) -> bool:
    """Return whether an evaluation file reached the expected final update."""
    if not path.is_file():
        return False
    evaluations = load_evaluations(path)
    return bool(evaluations and int(evaluations[-1]["update"]) >= expected_final_update)


def read_manifest(path: Path) -> list[tuple[str, str, float]]:
    """Read game slug, ALE environment ID, and paper score rows."""
    rows: list[tuple[str, str, float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 3:
            raise ValueError(f"invalid manifest row at {path}:{line_number}")
        slug, environment_id, paper_score = fields
        rows.append((slug, environment_id, float(paper_score)))
    if not rows:
        raise ValueError(f"game manifest is empty: {path}")
    return rows


def summarize(
    run_root: Path,
    *,
    experiment_name: str,
    expected_final_update: int,
) -> Path:
    """Write an atomic CSV summary for every game in the run manifest."""
    rows: list[dict[str, object]] = []
    for slug, environment_id, paper_score in read_manifest(run_root / "games.tsv"):
        evaluation_path = (
            run_root / slug / experiment_name / "evaluations" / "agent_evaluations.json"
        )
        row: dict[str, object] = {
            "game": slug,
            "environment": environment_id,
            "status": "pending",
            "final_update": "",
            "atariagent_mean": "",
            "atariagent_median": "",
            "atariagent_episode_std": "",
            "efficientzero_paper": paper_score,
            "score_ratio": "",
            "evaluation_path": evaluation_path,
        }
        if evaluation_path.is_file():
            evaluations = load_evaluations(evaluation_path)
            if evaluations:
                final = evaluations[-1]
                final_update = int(final["update"])
                mean = float(final["mean"])
                row.update(
                    status=(
                        "complete"
                        if final_update >= expected_final_update
                        else "partial"
                    ),
                    final_update=final_update,
                    atariagent_mean=mean,
                    atariagent_median=float(final["median"]),
                    atariagent_episode_std=float(final["std"]),
                    score_ratio=mean / paper_score,
                )
        rows.append(row)

    summary_path = run_root / "results.csv"
    temporary_path = summary_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(summary_path)
    return summary_path


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser(
        "summarize", help="write results.csv for a multi-game run"
    )
    summarize_parser.add_argument("run_root", type=Path)
    summarize_parser.add_argument("--experiment-name", default="puct-default")
    summarize_parser.add_argument(
        "--expected-final-update", type=positive_integer, default=120000
    )

    complete_parser = subparsers.add_parser(
        "is-complete", help="check whether an evaluation reached its final update"
    )
    complete_parser.add_argument("evaluation_path", type=Path)
    complete_parser.add_argument(
        "--expected-final-update", type=positive_integer, default=120000
    )
    return parser.parse_args()


def main() -> int:
    """Run the selected result operation."""
    args = parse_args()
    if args.command == "summarize":
        summary_path = summarize(
            args.run_root,
            experiment_name=args.experiment_name,
            expected_final_update=args.expected_final_update,
        )
        print(f"Updated result summary: {summary_path}")
        return 0
    if args.command == "is-complete":
        return 0 if is_complete(args.evaluation_path, args.expected_final_update) else 1
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
