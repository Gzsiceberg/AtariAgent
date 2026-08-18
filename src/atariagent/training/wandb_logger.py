"""Optional Weights & Biases logging for agent training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from atariagent.evaluation import EvaluationStats

from .config import WandbConfig, environment_slug
from .learner import TrainMetrics


class _WandbRun(Protocol):
    """Subset of the W&B run API used by training."""

    def define_metric(self, name: str, *, step_metric: str | None = None) -> object: ...

    def log(self, data: Mapping[str, object]) -> None: ...

    def finish(self, exit_code: int | None = None) -> None: ...


def wandb_run_name(environment_id: str, commit_hash: str) -> str:
    """Build a W&B run name from the Atari game and repository commit."""
    if not commit_hash:
        raise ValueError("commit_hash must not be empty")
    game_name = environment_slug(environment_id).removesuffix("-v5")
    return f"{game_name}_{commit_hash}"


class WandbLogger:
    """Log scalar training, self-play, and evaluation metrics when enabled."""

    def __init__(self, run: _WandbRun | None = None) -> None:
        self._run = run
        if run is not None:
            for namespace, step_name in (
                ("train", "train/update"),
                ("self_play", "self_play/transitions"),
                ("eval", "eval/update"),
            ):
                run.define_metric(step_name)
                run.define_metric(f"{namespace}/*", step_metric=step_name)

    @classmethod
    def initialize(
        cls,
        config: WandbConfig,
        *,
        run_name: str,
        run_config: Mapping[str, object],
    ) -> WandbLogger:
        """Initialize a W&B run, or return a no-op logger when disabled."""
        if not config.enabled:
            return cls()

        import wandb

        run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=run_name,
            tags=config.tags,
            job_type="train",
            config=dict(run_config),
        )
        if run is None:
            raise RuntimeError("wandb.init() did not create a run")
        return cls(run)

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log_training(self, metrics: TrainMetrics, *, update: int) -> None:
        """Log optimizer loss and gradient metrics for one update."""
        if self._run is None:
            return
        self._run.log(
            {
                "train/update": update,
                "train/loss": metrics.loss.item(),
                "train/policy_loss": metrics.policy_loss.item(),
                "train/value_loss": metrics.value_loss.item(),
                "train/reward_loss": metrics.reward_loss.item(),
                "train/consistency_loss": metrics.consistency_loss.item(),
                "train/gradient_norm": metrics.gradient_norm.item(),
            }
        )

    def log_self_play(
        self,
        stats: EvaluationStats,
        *,
        recent_rewards: Sequence[float],
        latest_reward: float,
        transitions: int,
        iteration: int,
        new_episodes: int,
        total_episodes: int,
    ) -> None:
        """Log rolling full-game self-play reward statistics."""
        if self._run is None:
            return
        self._run.log(
            {
                "self_play/transitions": transitions,
                "self_play/iteration": iteration,
                "self_play/new_episodes": new_episodes,
                "self_play/total_episodes": total_episodes,
                "self_play/reward_window": len(recent_rewards),
                "self_play/reward_mean_100": stats.mean,
                "self_play/reward_median_100": stats.median,
                "self_play/reward_std_100": stats.std,
                "self_play/reward_max_100": max(recent_rewards),
                "self_play/reward_latest": latest_reward,
            }
        )

    def log_evaluation(self, stats: EvaluationStats, *, update: int) -> None:
        """Log checkpoint evaluation reward statistics."""
        if self._run is None:
            return
        self._run.log(
            {
                "eval/update": update,
                "eval/episodes": len(stats.rewards),
                "eval/reward_mean": stats.mean,
                "eval/reward_median": stats.median,
                "eval/reward_std": stats.std,
                "eval/reward_min": min(stats.rewards),
                "eval/reward_max": max(stats.rewards),
            }
        )

    def finish(self, *, exit_code: int = 0) -> None:
        """Finish the active W&B run."""
        if self._run is not None:
            self._run.finish(exit_code=exit_code)
