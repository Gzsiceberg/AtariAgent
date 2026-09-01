"""Optional Weights & Biases logging for agent training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from atariagent.evaluation import EvaluationStats
from atariagent.selfplay import BehaviorPolicyMetrics

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
                ("self_play", "self_play/update"),
                ("behavior", "behavior/update"),
                ("eval", "eval/update"),
                ("reanalysis", "reanalysis/update"),
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

        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "Weights & Biases logging requires the optional dependency; "
                "install it with `uv sync --extra wandb`"
            ) from error
        if not hasattr(wandb, "init"):
            raise RuntimeError(
                "Weights & Biases logging requires the optional dependency; "
                "install it with `uv sync --extra wandb`"
            )

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

    def log_training(
        self,
        metrics: TrainMetrics,
        *,
        update: int,
        policy_roots_requested: int = 0,
        policy_roots_searched: int = 0,
        cache_hits: int = 0,
        cache_target_age_mean: float = 0.0,
        cache_target_age_max: int = 0,
    ) -> None:
        """Log optimizer, policy, and optional reanalysis diagnostics."""
        if self._run is None:
            return
        data: dict[str, object] = {
            "train/update": update,
            "train/loss": metrics.loss.item(),
            "train/policy_loss": metrics.policy_loss.item(),
            "train/value_loss": metrics.value_loss.item(),
            "train/reward_loss": metrics.reward_loss.item(),
            "train/consistency_loss": metrics.consistency_loss.item(),
            "train/gradient_norm": metrics.gradient_norm.item(),
            "train/search_target_entropy": metrics.search_target_entropy.item(),
            "train/network_policy_entropy": metrics.network_policy_entropy.item(),
            "train/policy_kl_divergence": metrics.policy_kl_divergence.item(),
            "train/search_target_max_probability": (
                metrics.search_target_max_probability.item()
            ),
            "train/network_policy_max_probability": (
                metrics.network_policy_max_probability.item()
            ),
            "train/search_target_effective_actions": (
                metrics.search_target_effective_actions.item()
            ),
            "train/network_policy_effective_actions": (
                metrics.network_policy_effective_actions.item()
            ),
        }
        if policy_roots_requested > 0:
            data.update(
                {
                    "reanalysis/update": update,
                    "reanalysis/cache_hit_rate": (
                        cache_hits / policy_roots_requested
                    ),
                    "reanalysis/cache_target_age_mean_updates": (
                        cache_target_age_mean
                    ),
                    "reanalysis/cache_target_age_max_updates": (
                        cache_target_age_max
                    ),
                    "reanalysis/policy_roots_requested": (
                        policy_roots_requested
                    ),
                    "reanalysis/policy_roots_searched": (
                        policy_roots_searched
                    ),
                }
            )
        self._run.log(data)

    def log_behavior_policy(
        self,
        metrics: BehaviorPolicyMetrics,
        *,
        total_transitions: int,
        update: int,
    ) -> None:
        """Log PUCT's categorical behavior policy after action temperature."""
        if self._run is None:
            return
        self._run.log(
            {
                "behavior/total_transitions": total_transitions,
                "behavior/update": update,
                "behavior/policy_entropy": metrics.entropy,
                "behavior/max_action_probability": metrics.max_probability,
                "behavior/effective_action_count": (
                    metrics.effective_action_count
                ),
            }
        )

    def log_self_play(
        self,
        stats: EvaluationStats,
        *,
        recent_rewards: Sequence[float],
        total_episodes: int,
        update: int,
    ) -> None:
        """Log rolling full-game self-play reward statistics."""
        if self._run is None:
            return
        self._run.log(
            {
                "self_play/total_episodes": total_episodes,
                "self_play/update": update,
                "self_play/reward_mean_10": stats.mean,
                "self_play/reward_median_10": stats.median,
                "self_play/reward_std_10": stats.std,
                "self_play/reward_min_10": min(recent_rewards),
                "self_play/reward_max_10": max(recent_rewards),
            }
        )

    def log_evaluation(self, stats: EvaluationStats, *, update: int) -> None:
        """Log checkpoint evaluation reward statistics."""
        if self._run is None:
            return
        self._run.log(
            {
                "eval/update": update,
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
