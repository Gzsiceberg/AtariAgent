import sys
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

from atariagent.evaluation import EvaluationStats
from atariagent.selfplay import BehaviorPolicyMetrics
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


def test_self_play_truncations_count_only_full_episode_boundaries() -> None:
    run = FakeRun()
    logger = WandbLogger(run)

    def trajectory(*, done: bool, truncated: bool = False):
        return SimpleNamespace(full_episode_done=done, truncated=truncated)

    logger.log_self_play_truncations(
        [trajectory(done=False), trajectory(done=False, truncated=True)],
        update=1,
    )
    assert run.logged == []
    logger.log_self_play_truncations(
        [
            trajectory(done=True, truncated=True),
            trajectory(done=False, truncated=True),  # Overlapping lookahead tail.
            trajectory(done=True),
        ],
        update=2,
    )
    assert run.logged[-1] == {
        "self_play/update": 2,
        "self_play/time_limit_truncated_episodes": 1,
        "self_play/time_limit_truncation_rate": 0.5,
    }
    logger.log_self_play_truncations([trajectory(done=True)], update=3)
    assert run.logged[-1]["self_play/time_limit_truncated_episodes"] == 1
    assert run.logged[-1]["self_play/time_limit_truncation_rate"] == pytest.approx(1 / 3)
    WandbLogger().log_self_play_truncations(
        [trajectory(done=True, truncated=True)], update=1
    )


def make_train_metrics() -> TrainMetrics:
    return TrainMetrics(
        loss=torch.tensor(6.0),
        policy_loss=torch.tensor(11.0),
        value_loss=torch.tensor(12.0),
        reward_loss=torch.tensor(13.0),
        consistency_loss=torch.tensor(-0.5),
        gradient_norm=torch.tensor(5.0),
        search_target_entropy=torch.tensor(0.1),
        network_policy_entropy=torch.tensor(0.2),
        policy_kl_divergence=torch.tensor(0.3),
        search_target_max_probability=torch.tensor(0.4),
        network_policy_max_probability=torch.tensor(0.5),
        search_target_effective_actions=torch.tensor(1.1),
        network_policy_effective_actions=torch.tensor(1.2),
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
            "train/search_target_entropy": pytest.approx(0.1),
            "train/network_policy_entropy": pytest.approx(0.2),
            "train/policy_kl_divergence": pytest.approx(0.3),
            "train/search_target_max_probability": pytest.approx(0.4),
            "train/network_policy_max_probability": pytest.approx(0.5),
            "train/search_target_effective_actions": pytest.approx(1.1),
            "train/network_policy_effective_actions": pytest.approx(1.2),
        }
    ]


@pytest.mark.parametrize("ready_count", [0, 2])
def test_wandb_logger_emits_prefetch_ready_count(ready_count: int) -> None:
    run = FakeRun()
    logger = WandbLogger(run)

    logger.log_training(
        make_train_metrics(), update=10, prefetch_ready_batches=ready_count
    )

    assert run.logged[0]["train/update"] == 10
    assert run.logged[0]["train/prefetch_ready_batches"] == ready_count


def test_wandb_logger_emits_reanalysis_metrics() -> None:
    run = FakeRun()
    logger = WandbLogger(run)

    logger.log_training(
        make_train_metrics(),
        update=10,
        policy_roots_requested=100,
        policy_roots_searched=25,
        cache_hits=75,
        cache_target_age_mean=12.5,
        cache_target_age_max=30,
    )

    assert ("reanalysis/*", "reanalysis/update") in run.defined_metrics
    assert run.logged[0]["reanalysis/update"] == 10
    assert run.logged[0]["reanalysis/cache_hit_rate"] == pytest.approx(0.75)
    assert run.logged[0]["reanalysis/cache_target_age_mean_updates"] == 12.5
    assert run.logged[0]["reanalysis/cache_target_age_max_updates"] == 30


def test_wandb_logger_emits_self_play_and_evaluation_rewards() -> None:
    run = FakeRun()
    logger = WandbLogger(run)
    stats = EvaluationStats.from_rewards((1.0, 2.0, 6.0))

    logger.log_behavior_policy(
        BehaviorPolicyMetrics(
            entropy=0.5,
            max_probability=0.75,
            effective_action_count=1.5,
            root_count=8,
        ),
        total_transitions=400,
        update=100,
    )
    logger.log_self_play(
        stats,
        recent_rewards=stats.rewards,
        total_episodes=3,
        update=100,
    )
    logger.log_evaluation(stats, update=100)
    logger.finish(exit_code=0)

    behavior, self_play, evaluation = run.logged
    assert ("behavior/*", "behavior/update") in run.defined_metrics
    assert behavior["behavior/update"] == 100
    assert behavior["behavior/policy_entropy"] == pytest.approx(0.5)
    assert behavior["behavior/max_action_probability"] == pytest.approx(0.75)
    assert behavior["behavior/effective_action_count"] == pytest.approx(1.5)
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
