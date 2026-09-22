import sys
from collections.abc import Mapping
from types import SimpleNamespace

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
        policy_loss=torch.tensor(11.0),
        value_loss=torch.tensor(12.0),
        reward_loss=torch.tensor(13.0),
        consistency_loss=torch.tensor(-0.5),
        behavior_regularization_loss=torch.tensor(0.0),
        gradient_norm=torch.tensor(5.0),
        search_target_entropy=torch.tensor(0.1),
        search_target_max_probability=torch.tensor(0.4),
        search_target_effective_actions=torch.tensor(1.1),
        importance_weight_mean=torch.tensor(0.5),
        importance_weight_ess_fraction=torch.tensor(0.6),
        representation_feature_variance=torch.tensor(0.002),
        dynamics_feature_variance=torch.tensor(0.003),
        learning_rate=0.2,
        priorities=torch.ones(2),
    )


def test_wandb_run_uses_game_and_optional_suffix_as_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeRun()
    initialization: dict[str, object] = {}

    def fake_init(**kwargs: object) -> FakeRun:
        initialization.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    assert wandb_run_name("ALE/MsPacman-v5") == "MsPacman"
    logger = WandbLogger.initialize(
        WandbConfig(enabled=True),
        run_name=wandb_run_name("ALE/MsPacman-v5", "mixed-threshold-20000"),
        run_config={"seed": 2},
    )

    assert logger.enabled
    assert initialization["name"] == "MsPacman_mixed-threshold-20000"


def test_wandb_logger_emits_training_metrics() -> None:
    run = FakeRun()
    logger = WandbLogger(run)

    logger.log_training(make_train_metrics(), update=10)

    assert ("train/*", "train/update") in run.defined_metrics
    assert run.logged == [
        {
            "train/update": 10,
            "train/loss": 6.0,
            "train/policy_loss": 11.0,
            "train/value_loss": 12.0,
            "train/reward_loss": 13.0,
            "train/consistency_loss": -0.5,
            "train/gradient_norm": 5.0,
            "train/importance_weight_mean": pytest.approx(0.5),
            "train/importance_weight_ess_fraction": pytest.approx(0.6),
            "train/representation_feature_variance": pytest.approx(0.002),
            "train/dynamics_feature_variance": pytest.approx(0.003),
            "train/search_target_entropy": pytest.approx(0.1),
            "train/search_target_max_probability": pytest.approx(0.4),
            "train/search_target_effective_actions": pytest.approx(1.1),
        }
    ]


def test_wandb_logger_emits_reanalysis_metrics() -> None:
    run = FakeRun()
    logger = WandbLogger(run)

    logger.log_training(
        make_train_metrics(),
        update=10,
        policy_roots_requested=100,
        policy_roots_searched=25,
        cache_hits=75,
    )

    assert ("reanalysis/*", "reanalysis/update") in run.defined_metrics
    assert run.logged[0]["reanalysis/update"] == 10
    assert run.logged[0]["reanalysis/cache_hit_rate"] == pytest.approx(0.75)


def test_wandb_logger_emits_current_self_play_rewards() -> None:
    run = FakeRun()
    logger = WandbLogger(run)
    stats = EvaluationStats.from_rewards((1.0, 2.0, 6.0))

    logger.log_current_self_play(stats.rewards, update=0)

    assert run.logged == [{
        "self_play/update": 0,
        "self_play/current_reward_env_0": 1.0,
        "self_play/current_reward_env_1": 2.0,
        "self_play/current_reward_env_2": 6.0,
    }]
    WandbLogger(None).log_current_self_play(stats.rewards, update=0)


def test_wandb_logger_emits_self_play_and_evaluation_rewards() -> None:
    run = FakeRun()
    logger = WandbLogger(run)
    stats = EvaluationStats.from_rewards((1.0, 2.0, 6.0))

    logger.log_self_play(
        stats,
        recent_rewards=stats.rewards,
        total_episodes=3,
        update=100,
    )
    logger.log_evaluation(stats, update=100)
    logger.finish(exit_code=0)

    self_play, evaluation = run.logged
    assert ("self_play/*", "self_play/update") in run.defined_metrics
    assert self_play["self_play/update"] == 100
    assert self_play["self_play/reward_mean_10"] == pytest.approx(3.0)
    assert self_play["self_play/reward_min_10"] == 1.0
    assert self_play["self_play/reward_max_10"] == 6.0
    for removed_metric in (
        "self_play/transitions",
        "self_play/iteration",
        "self_play/new_episodes",
        "self_play/reward_window",
        "self_play/reward_mean_100",
        "self_play/reward_min_100",
        "self_play/reward_max_100",
        "self_play/reward_latest",
    ):
        assert removed_metric not in self_play
    assert ("eval/*", "eval/update") in run.defined_metrics
    assert evaluation["eval/update"] == 100
    assert "eval/episodes" not in evaluation
    assert evaluation["eval/reward_mean"] == pytest.approx(3.0)
    assert evaluation["eval/reward_min"] == 1.0
    assert evaluation["eval/reward_max"] == 6.0
    assert run.exit_code == 0


def test_enabled_wandb_logger_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(RuntimeError, match="uv sync --extra wandb"):
        WandbLogger.initialize(
            WandbConfig(enabled=True),
            run_name="missing-dependency",
            run_config={},
        )


def test_disabled_wandb_logger_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = WandbLogger.initialize(
        WandbConfig(),
        run_name="disabled",
        run_config={},
    )

    logger.log_training(make_train_metrics(), update=1)
    logger.log_evaluation(EvaluationStats.from_rewards((1.0,)), update=1)
    logger.finish(exit_code=1)

    assert not logger.enabled
