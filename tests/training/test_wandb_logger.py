from collections.abc import Mapping

import pytest
import torch

from atariagent.evaluation import EvaluationStats
from atariagent.training import TrainMetrics, WandbLogger, wandb_run_name
from atariagent.training.config import WandbConfig


class FakeRun:
    def __init__(self) -> None:
        self.defined_metrics: list[tuple[str, str | None]] = []
        self.logged: list[dict[str, object]] = []
        self.exit_code: int | None = None

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        self.defined_metrics.append((name, step_metric))

    def log(self, data: Mapping[str, object]) -> None:
        self.logged.append(dict(data))

    def finish(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code


def make_train_metrics() -> TrainMetrics:
    return TrainMetrics(
        loss=torch.tensor(6.0),
        policy_loss=torch.tensor(1.0),
        value_loss=torch.tensor(2.0),
        reward_loss=torch.tensor(3.0),
        consistency_loss=torch.tensor(4.0),
        gradient_norm=torch.tensor(5.0),
        learning_rate=0.2,
        priorities=torch.ones(2),
    )


def test_wandb_run_uses_game_and_repository_commit_as_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wandb

    run = FakeRun()
    initialization: dict[str, object] = {}

    def fake_init(**kwargs: object) -> FakeRun:
        initialization.update(kwargs)
        return run

    monkeypatch.setattr(wandb, "init", fake_init)
    logger = WandbLogger.initialize(
        WandbConfig(),
        run_name=wandb_run_name("ALE/MsPacman-v5", "0123456789abcdef"),
        run_config={"seed": 2},
    )

    assert logger.enabled
    assert initialization["name"] == "MsPacman_0123456789abcdef"


def test_wandb_logger_emits_training_metrics() -> None:
    run = FakeRun()
    logger = WandbLogger(run)

    logger.log_training(make_train_metrics(), update=10)

    assert ("train/*", "train/update") in run.defined_metrics
    assert run.logged == [
        {
            "train/update": 10,
            "train/loss": 6.0,
            "train/policy_loss": 1.0,
            "train/value_loss": 2.0,
            "train/reward_loss": 3.0,
            "train/consistency_loss": 4.0,
            "train/gradient_norm": 5.0,
        }
    ]


def test_wandb_logger_emits_self_play_and_evaluation_rewards() -> None:
    run = FakeRun()
    logger = WandbLogger(run)
    stats = EvaluationStats.from_rewards((1.0, 2.0, 6.0))

    logger.log_self_play(
        stats,
        recent_rewards=stats.rewards,
        latest_reward=6.0,
        transitions=400,
        iteration=2,
        new_episodes=1,
        total_episodes=3,
    )
    logger.log_evaluation(stats, update=100)
    logger.finish(exit_code=0)

    self_play, evaluation = run.logged
    assert self_play["self_play/reward_mean_100"] == pytest.approx(3.0)
    assert "self_play/reward_min_100" not in self_play
    assert self_play["self_play/reward_max_100"] == 6.0
    assert evaluation["eval/reward_mean"] == pytest.approx(3.0)
    assert evaluation["eval/reward_min"] == 1.0
    assert evaluation["eval/reward_max"] == 6.0
    assert run.exit_code == 0


def test_disabled_wandb_logger_is_a_no_op() -> None:
    logger = WandbLogger()

    logger.log_training(make_train_metrics(), update=1)
    logger.log_evaluation(EvaluationStats.from_rewards((1.0,)), update=1)
    logger.finish(exit_code=1)

    assert not logger.enabled
