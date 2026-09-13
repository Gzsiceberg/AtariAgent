"""Regression tests for best-evaluation summaries."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_game_experiments.py"
spec = importlib.util.spec_from_file_location("summarize_game_experiments", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.mark.parametrize("final_update,status", [(120000, "complete"), (60000, "partial")])
def test_summary_selects_best_mean_and_normalizes(tmp_path, final_update, status):
    (tmp_path / "games.tsv").write_text("asterix|ALE/Asterix-v5\nalien|ALE/Alien-v5\n")
    paper = tmp_path / "paper.csv"
    paper.write_text(
        "game,random,human,efficientzero_v1,efficientzero_v2\n"
        "asterix,10,110,50,60\n"
        "alien,0,100,50,60\n"
        "normed-mean,0,1,1.945,2.428\n"
        "normed-median,0,1,1.09,1.286\n"
    )
    path = tmp_path / "asterix/default/evaluations/agent_evaluations.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"evaluations": [
        {"update": 20000, "mean": 210, "median": 190, "std": 12},
        {"update": final_update, "mean": 110, "median": 100, "std": 5},
    ]}))
    output = summary.summarize(
        tmp_path, experiment_name="default", expected_final_update=120000,
        paper_scores_path=paper,
    )
    with output.open() as source:
        rows = {row["game"]: row for row in csv.DictReader(source)}
    game = rows["asterix"]
    assert game["status"] == status
    assert int(game["final_update"]) == final_update
    assert int(game["best_update"]) == 20000
    assert float(game["atariagent_mean"]) == 210
    assert float(game["atariagent_median"]) == 190
    assert float(game["atariagent_episode_std"]) == 12
    assert float(game["human_normalized_score"]) == 2
    assert float(game["score_ratio"]) == 4.2
    assert rows["alien"]["status"] == "pending"
    for name in ("normed-mean", "normed-median"):
        assert float(rows[name]["human_normalized_score"]) == 2
        assert rows[name]["aggregate_game_count"] == "1"
