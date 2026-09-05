from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from atariagent.training.config import (
    AugmentationConfig,
    EvaluationConfig,
    LossConfig,
    ReanalysisConfig,
    ReplayConfig,
    SelfPlayConfig,
    TrainingConfig,
    WandbConfig,
    checkpoint_path_for_environment,
    environment_slug,
    final_evaluation_max_episode_steps,
    linear_priority_beta,
    next_collection_vector_steps,
    proportional_training_update,
    register_train_agent_config,
    scheduled_cache_clear_interval,
    target_network_update_due,
    visit_softmax_temperature,
)


def test_output_paths_are_derived_from_environment_id() -> None:
    assert environment_slug("ALE/Pong-v5") == "Pong-v5"
    assert environment_slug("Pong-v5") == "Pong-v5"
    assert checkpoint_path_for_environment("ALE/Breakout-v5") == (
        "checkpoints/Breakout-v5/agent_latest.pt"
    )

    with pytest.raises(ValueError, match="valid final component"):
        environment_slug("ALE/")


def test_replay_uses_efficientzero_v1_per_defaults() -> None:
    config = ReplayConfig()

    assert config.per_mode == "v1"
    assert config.priority_alpha == pytest.approx(0.6)
    assert config.priority_beta_initial == pytest.approx(0.4)
    assert config.priority_beta_final == pytest.approx(1.0)
    assert linear_priority_beta(0, 120_000) == pytest.approx(0.4)
    assert linear_priority_beta(100_000, 120_000) == pytest.approx(0.9)
    assert linear_priority_beta(120_000, 120_000) == pytest.approx(1.0)


def test_replay_per_mode_can_select_v2() -> None:
    register_train_agent_config()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="train_agent",
            overrides=["replay.per_mode=v2"],
        )

    assert config.replay.per_mode == "v2"


def test_environment_time_limit_mode_can_be_overridden() -> None:
    register_train_agent_config()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        default = compose(config_name="train_agent")
        per_life = compose(
            config_name="train_agent",
            overrides=["environment.time_limit_mode=episodic_life"],
        )

    assert default.environment.time_limit_mode == "full_game"
    assert per_life.environment.time_limit_mode == "episodic_life"


def test_augmentation_uses_efficientzero_atari_defaults() -> None:
    config = AugmentationConfig()

    assert config.enabled
    assert config.transforms == ["shift", "intensity"]
    assert config.shift_delta == 4
    assert config.intensity_scale == pytest.approx(0.05)


def test_self_play_defaults_to_50_simulation_puct() -> None:
    config = SelfPlayConfig()

    assert config.search_algorithm == "puct"
    assert config.num_simulations == 50
    assert config.root_exploration_fraction == pytest.approx(0.25)
    assert config.num_envs * config.steps_per_iteration == 100


def test_search_presets_select_their_simulation_budgets() -> None:
    register_train_agent_config()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        puct = compose(config_name="train_agent")
        custom_puct = compose(
            config_name="train_agent",
            overrides=["self_play.root_exploration_fraction=0.4"],
        )
        gumbel = compose(config_name="train_agent", overrides=["search=gumbel"])

    assert puct.self_play.search_algorithm == "puct"
    assert puct.self_play.num_simulations == 50
    assert puct.self_play.root_exploration_fraction == pytest.approx(0.25)
    assert puct.self_play.steps_per_iteration == 25
    assert puct.training.updates_per_iteration == 100
    assert custom_puct.self_play.root_exploration_fraction == pytest.approx(0.4)
    assert puct.reanalysis.initial_cache_clear_interval == 100
    assert gumbel.self_play.search_algorithm == "gumbel"
    assert gumbel.self_play.num_simulations == 16
    assert gumbel.reanalysis.initial_cache_clear_interval == 100


def test_resume_evaluation_is_disabled_by_default_and_can_be_enabled() -> None:
    config = EvaluationConfig()
    register_train_agent_config()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        resumed = compose(
            config_name="train_agent",
            overrides=[
                "checkpoint.resume_pre_final_path=/tmp/agent_pre_final.pt",
                "evaluation.evaluate_on_resume=true",
            ],
        )

    assert not config.evaluate_on_resume
    assert resumed.evaluation.evaluate_on_resume


def test_wandb_is_disabled_by_default() -> None:
    config = WandbConfig()

    assert not config.enabled
    assert config.project == "AtariAgent"
    assert config.entity is None
    assert config.name is None
    assert config.tags == []


def test_consistency_loss_uses_efficientzero_defaults() -> None:
    config = LossConfig()

    assert config.consistency_weight == pytest.approx(5.0)


def test_target_network_uses_efficientzero_hard_copy_interval() -> None:
    config = TrainingConfig()
    assert config.value_target == "mixed"
    assert config.mixed_value_start_step == 30_000
    assert config.mixed_value_threshold == 20_000
    assert config.batch_max_in_flight == 3
    assert config.batch_ready_prefetch == 2
    assert config.batch_worker_timeout_seconds == pytest.approx(600.0)
    assert config.optimizer == "adam"
    assert config.learning_rate == pytest.approx(0.001)
    assert config.lr_warmup_steps == 1_000
    assert config.lr_decay_rate == pytest.approx(0.1)
    assert config.lr_decay_steps == 100_000
    assert config.steps == 100_000
    assert config.final_steps == 20_000
    assert config.updates_per_iteration == 100
    assert config.compile_mode == "max-autotune"
    assert config.progress_mode == "auto"
    assert config.progress_interval_seconds == pytest.approx(0.1)


def test_cudnn_autotuning_defaults_off_and_can_be_enabled() -> None:
    assert TrainingConfig().cudnn_benchmark is False
    register_train_agent_config()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        default = compose(config_name="train_agent")
        enabled = compose(
            config_name="train_agent",
            overrides=["training.cudnn_benchmark=true"],
        )

    assert default.training.cudnn_benchmark is False
    assert enabled.training.cudnn_benchmark is True
    assert enabled.training.deterministic == default.training.deterministic
    assert enabled.training.precision == default.training.precision


def test_progress_can_be_forced_for_redirected_batch_output() -> None:
    register_train_agent_config()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="train_agent",
            overrides=[
                "training.progress_mode=always",
                "training.progress_interval_seconds=10",
            ],
        )

    assert config.training.progress_mode == "always"
    assert config.training.progress_interval_seconds == pytest.approx(10.0)


def test_reanalysis_uses_target_network_defaults() -> None:
    config = ReanalysisConfig()

    assert config.policy_chunk_size == 768
    assert config.cache_targets
    assert config.initial_cache_clear_interval == 100
    assert config.cache_target_ttl == 200
    assert config.prefetch_batches == 2
    assert config.timeout_seconds == pytest.approx(600.0)
    assert config.worker_num_threads == 4
    assert config.target_update_interval == 1_000


def test_target_network_uses_fixed_update_interval() -> None:
    interval = ReanalysisConfig().target_update_interval
    last_update = 0
    due_updates = []
    for update in range(1, 3_001):
        if target_network_update_due(
            update,
            last_update=last_update,
            interval=interval,
        ):
            due_updates.append(update)
            last_update = update

    assert due_updates == [1_000, 2_000, 3_000]


def test_cache_clear_interval_ramps_over_first_half_of_training() -> None:
    intervals = [
        scheduled_cache_clear_interval(
            update,
            ramp_steps=5_000,
            initial_interval=100,
            final_interval=1_000,
        )
        for update in (0, 2_500, 5_000, 10_000)
    ]
    assert intervals == [100, 550, 1_000, 1_000]


def test_final_evaluation_uses_efficientzero_v1_raw_frame_horizon() -> None:
    assert final_evaluation_max_episode_steps(4) == 27_000

    with pytest.raises(ValueError, match="frame_skip"):
        final_evaluation_max_episode_steps(0)


def test_collection_steps_stop_at_exact_transition_budget() -> None:
    collected = 0
    while collected < 100_000:
        vector_steps = next_collection_vector_steps(
            collected, 100_000, num_envs=4, max_vector_steps=100
        )
        collected += vector_steps * 4

    assert collected == 100_000
    assert next_collection_vector_steps(100_000, 100_000, 4, 100) == 0


def test_training_update_tracks_collection_progress_after_warmup() -> None:
    assert proportional_training_update(0, 100_000, 100_000) == 0
    assert proportional_training_update(2_400, 100_000, 100_000) == 2_400
    assert proportional_training_update(50_000, 100_000, 100_000) == 50_000
    assert proportional_training_update(100_000, 100_000, 100_000) == 100_000
    assert proportional_training_update(25, 100, 200) == 50


def test_proportional_training_update_validates_budgets() -> None:
    with pytest.raises(TypeError, match="collected_transitions"):
        proportional_training_update(1.0, 100, 100)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be positive"):
        proportional_training_update(0, 0, 100)
    with pytest.raises(ValueError, match="between zero"):
        proportional_training_update(101, 100, 100)


def test_visit_temperature_uses_efficientzero_v1_collection_schedule() -> None:
    collection_steps = 100_000

    assert visit_softmax_temperature(0, collection_steps) == 1.0
    assert visit_softmax_temperature(49_999, collection_steps) == 1.0
    assert visit_softmax_temperature(50_000, collection_steps) == 0.5
    assert visit_softmax_temperature(74_999, collection_steps) == 0.5
    assert visit_softmax_temperature(75_000, collection_steps) == 0.25
    assert visit_softmax_temperature(100_000, collection_steps) == 0.25


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 1.0), (59_999, 1.0), (60_000, 0.5), (89_999, 0.5), (90_000, 0.25)],
)
def test_visit_temperature_can_include_final_steps(step: int, expected: float) -> None:
    assert visit_softmax_temperature(
        step, 100_000, final_steps=20_000, horizon="total_steps"
    ) == expected


def test_visit_temperature_default_excludes_final_steps() -> None:
    assert TrainingConfig().visit_softmax_temperature_horizon == "collect_steps"
    assert visit_softmax_temperature(50_000, 100_000, final_steps=20_000) == 0.5


@pytest.mark.parametrize("horizon", ["collect_steps", "total_steps"])
def test_visit_temperature_horizon_can_be_configured(horizon: str) -> None:
    register_train_agent_config()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="train_agent",
            overrides=[f"training.visit_softmax_temperature_horizon={horizon}"],
        )
    assert config.training.visit_softmax_temperature_horizon == horizon


def test_visit_temperature_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        visit_softmax_temperature(0, 0)
    with pytest.raises(ValueError, match="trained_steps"):
        visit_softmax_temperature(-1, 100)
    with pytest.raises(ValueError, match="final_steps"):
        visit_softmax_temperature(0, 100, final_steps=-1)
    with pytest.raises(ValueError, match="horizon"):
        visit_softmax_temperature(0, 100, horizon="invalid")
