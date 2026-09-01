#!/usr/bin/env python3
"""Summarize and inspect AtariAgent multi-game experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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


def read_manifest(path: Path) -> list[tuple[str, str]]:
    """Read game slug and ALE environment ID rows."""
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) not in {2, 3}:
            raise ValueError(f"invalid manifest row at {path}:{line_number}")
        # The optional third field is the V1 score embedded by older versions.
        # Ignore it in favor of the shared paper-score CSV.
        slug, environment_id = fields[:2]
        rows.append((slug, environment_id))
    if not rows:
        raise ValueError(f"game manifest is empty: {path}")
    return rows


def read_paper_scores(
    path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Read per-game scores and published normalized aggregate rows."""
    required_fields = {
        "game",
        "random",
        "human",
        "efficientzero_v1",
        "efficientzero_v2",
    }
    aggregate_names = {"normed-mean", "normed-median"}
    scores: dict[str, dict[str, float]] = {}
    aggregates: dict[str, dict[str, float]] = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                f"paper score CSV must contain {sorted(required_fields)}: {path}"
            )
        for line_number, row in enumerate(reader, start=2):
            slug = row["game"].strip()
            if not slug:
                raise ValueError(f"missing game slug at {path}:{line_number}")
            destination = aggregates if slug in aggregate_names else scores
            if slug in destination:
                raise ValueError(f"duplicate game slug {slug!r} at {path}:{line_number}")
            destination[slug] = {
                "random": float(row["random"]),
                "human": float(row["human"]),
                "efficientzero_v1": float(row["efficientzero_v1"]),
                "efficientzero_v2": float(row["efficientzero_v2"]),
            }
    if not scores:
        raise ValueError(f"paper score CSV has no game rows: {path}")
    missing_aggregates = aggregate_names - aggregates.keys()
    if missing_aggregates:
        raise ValueError(
            f"paper score CSV is missing aggregate rows {sorted(missing_aggregates)}: "
            f"{path}"
        )
    return scores, aggregates


def summarize(
    run_root: Path,
    *,
    experiment_name: str,
    expected_final_update: int,
    paper_scores_path: Path,
) -> Path:
    """Write an atomic CSV summary for every game in the run manifest."""
    rows: list[dict[str, object]] = []
    normalized_scores: list[float] = []
    paper_scores, paper_aggregates = read_paper_scores(paper_scores_path)
    for slug, environment_id in read_manifest(run_root / "games.tsv"):
        if slug not in paper_scores:
            raise ValueError(f"paper score CSV has no row for game {slug!r}")
        references = paper_scores[slug]
        random_score = references["random"]
        human_score = references["human"]
        efficientzero_v1 = references["efficientzero_v1"]
        normalization_range = human_score - random_score
        if normalization_range == 0:
            raise ValueError(f"human and random scores are equal for game {slug!r}")
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
            "human_normalized_score": "",
            "random_paper": random_score,
            "human_paper": human_score,
            "efficientzero_paper": efficientzero_v1,
            "efficientzero_paper_v2": references["efficientzero_v2"],
            "score_ratio": "",
            "aggregate_game_count": "",
            "evaluation_path": evaluation_path,
        }
        if evaluation_path.is_file():
            evaluations = load_evaluations(evaluation_path)
            if evaluations:
                final = evaluations[-1]
                final_update = int(final["update"])
                mean = float(final["mean"])
                normalized_score = (mean - random_score) / normalization_range
                normalized_scores.append(normalized_score)
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
                    human_normalized_score=normalized_score,
                    score_ratio=mean / efficientzero_v1,
                )
        rows.append(row)

    aggregate_functions = {
        "normed-mean": statistics.fmean,
        "normed-median": statistics.median,
    }
    for aggregate_name, aggregate_function in aggregate_functions.items():
        references = paper_aggregates[aggregate_name]
        rows.append(
            {
                "game": aggregate_name,
                "environment": "",
                "status": "aggregate",
                "final_update": "",
                "atariagent_mean": "",
                "atariagent_median": "",
                "atariagent_episode_std": "",
                "human_normalized_score": (
                    aggregate_function(normalized_scores) if normalized_scores else ""
                ),
                "random_paper": references["random"],
                "human_paper": references["human"],
                "efficientzero_paper": references["efficientzero_v1"],
                "efficientzero_paper_v2": references["efficientzero_v2"],
                "score_ratio": "",
                "aggregate_game_count": len(normalized_scores),
                "evaluation_path": "",
            }
        )

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
    summarize_parser.add_argument(
        "--paper-scores",
        type=Path,
        default=Path(__file__).with_name("atari_100k_paper_scores.csv"),
        help="CSV containing human and EfficientZero V1/V2 reference scores",
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
            paper_scores_path=args.paper_scores,
        )
        print(f"Updated result summary: {summary_path}")
        return 0
    if args.command == "is-complete":
        return 0 if is_complete(args.evaluation_path, args.expected_final_update) else 1
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
