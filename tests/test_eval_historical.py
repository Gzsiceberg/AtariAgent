"""CPU-only checks for the historical checkpoint evaluation launcher."""
import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_historical.py"
spec = importlib.util.spec_from_file_location("eval_historical", SCRIPT)
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def config(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    return OmegaConf.create({
        "episodes": 128, "num_envs": 16, "seed": 2000, "device": "cuda",
        "num_simulations": 50, "dry_run": True,
        "output_dir": str(tmp_path / "output"),
        "checkpoints": {"candidate": str(checkpoint)},
    })


def test_dry_run_does_not_create_outputs(tmp_path, capsys):
    cfg = config(tmp_path)
    launcher.main.__wrapped__(cfg)
    text = capsys.readouterr().out
    assert "--seed 2000" in text
    assert "--search-algorithm puct" in text
    assert "--output-json" in text
    assert not Path(cfg.output_dir).exists()


def test_existing_output_is_rejected(tmp_path):
    cfg = config(tmp_path)
    Path(cfg.output_dir).mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        launcher.main.__wrapped__(cfg)


def test_missing_checkpoint_is_rejected_before_output_creation(tmp_path):
    cfg = config(tmp_path)
    Path(cfg.checkpoints.candidate).unlink()
    with pytest.raises(FileNotFoundError):
        launcher.main.__wrapped__(cfg)
    assert not Path(cfg.output_dir).exists()


def test_summary_constant_rewards():
    stats = launcher.summarize({"rewards": [50000.0] * 8}, 8)
    assert stats["mean"] == 50000
    assert stats["mean_episode_bootstrap_95_ci"] == [50000, 50000]


@pytest.mark.parametrize("rewards", [[1], [1, float("nan")], [[1, 2]]])
def test_summary_rejects_incomplete_or_invalid_returns(rewards):
    with pytest.raises(ValueError):
        launcher.summarize({"rewards": rewards}, 2)
