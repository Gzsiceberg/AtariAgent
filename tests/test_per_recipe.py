"""Test the recipe launcher without starting training or contacting W&B."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_dir

from atariagent.training.config import register_train_agent_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_per_recipe.sh"


@pytest.fixture
def launcher(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(int(os.environ.get('FAIL_TRAINING', '0')))\n"
    )
    fake_uv.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    root = tmp_path / "output with spaces"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RUN_ROOT": str(root),
        "RUN_ID": "test",
        "CALLS": str(calls),
        "WANDB_ENABLED": "false",
        "DRY_RUN": "0",
    }

    def run(**overrides):
        return subprocess.run(
            ["bash", str(SCRIPT)], env={**env, **overrides},
            capture_output=True, text=True, check=False,
        )

    return run, calls, root


def test_training_commands_and_valid_hydra_recipe(launcher):
    run, calls, root = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    names = ["puct_v2_t200_no_ttl_mv5000"]
    assert len(commands) == len(names)
    for name, command in zip(names, commands, strict=True):
        assert (
            f"checkpoint.pre_final_snapshot_path={root / name / 'checkpoints/agent_pre_final.pt'}"
            in command
        )
        assert (root / name / "command.sh").is_file()
    args = commands[0]
    assert args[:3] == ["run", "python", "scripts/train_agent.py"]
    register_train_agent_config()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(config_name="train_agent", overrides=args[3:])
    assert config.self_play.search_algorithm == "puct"
    assert config.self_play.num_simulations == 50
    assert config.replay.per_mode == "v2"
    assert config.training.mixed_value_threshold == 5000
    assert config.training.final_steps == 20000
    assert config.checkpoint.pre_final_snapshot_path == str(
        root / "puct_v2_t200_no_ttl_mv5000/checkpoints/agent_pre_final.pt"
    )
    assert config.checkpoint.resume_pre_final_path is None
    assert config.checkpoint.final_interval == 5000
    assert config.evaluation.num_envs == 16
    assert config.reanalysis.cache_targets is True
    assert config.reanalysis.target_update_interval == 200
    assert config.reanalysis.cache_target_ttl == 0
    assert run().returncode != 0
    assert len(calls.read_text().splitlines()) == 1


def test_dry_run_has_no_side_effects(launcher):
    run, calls, root = launcher
    result = run(DRY_RUN="1")
    assert result.returncode == 0
    assert result.stdout.count("checkpoint.pre_final_snapshot_path=") == 1
    assert "checkpoint.pre_final_snapshot_path=null" not in result.stdout
    assert not calls.exists()
    assert not root.exists()


def test_training_failure_is_propagated(launcher):
    run, calls, _ = launcher
    result = run(FAIL_TRAINING="7")
    assert result.returncode == 7
    assert len(calls.read_text().splitlines()) == 1
    assert "Experiments complete" not in result.stdout
