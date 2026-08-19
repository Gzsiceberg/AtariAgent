import pytest

from atariagent.training.config import (
    AugmentationConfig,
    LossConfig,
    TrainingConfig,
    WandbConfig,
    checkpoint_path_for_environment,
    environment_slug,
    final_evaluation_max_episode_steps,
    next_collection_vector_steps,
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


def test_augmentation_uses_efficientzero_atari_defaults() -> None:
    config = AugmentationConfig()

    assert config.enabled
    assert config.transforms == ["shift", "intensity"]
    assert config.shift_delta == 4
    assert config.intensity_scale == pytest.approx(0.05)


def test_wandb_is_enabled_by_default() -> None:
    config = WandbConfig()

    assert config.enabled
    assert config.project == "AtariAgent"
    assert config.entity is None
    assert config.tags == []


def test_consistency_loss_uses_efficientzero_defaults() -> None:
    config = LossConfig()

    assert config.consistency_enabled
    assert config.consistency_weight == pytest.approx(5.0)


def test_target_network_uses_efficientzero_hard_copy_interval() -> None:
    config = TrainingConfig()
    assert config.value_target == "mixed"
    assert config.mixed_value_start_step == 30_000
    assert config.mixed_value_threshold == 5_000
    assert config.use_target_network_reanalysis
    assert not config.reanalysis_add_dirichlet_noise
    assert config.policy_reanalysis_chunk_size == 768
    assert config.cache_reanalyzed_targets
    assert config.reanalysis_prefetch_batches == 2
    assert config.batch_max_in_flight == 3
    assert config.batch_ready_prefetch == 2
    assert config.batch_worker_timeout_seconds == pytest.approx(600.0)
    assert config.reanalysis_timeout_seconds == pytest.approx(600.0)
    assert config.reanalysis_worker_num_threads == 4
    assert config.target_update_interval == 1_000
    assert config.initial_target_update_interval == 200
    assert config.initial_target_update_steps == 1_000
    assert config.optimizer == "adam"
    assert config.learning_rate == pytest.approx(0.001)
    assert config.lr_warmup_steps == 1_000
    assert config.lr_decay_rate == pytest.approx(0.1)
    assert config.lr_decay_steps == 100_000
    assert config.steps == 100_000
    assert config.final_steps == 20_000
    assert config.compile_mode == "max-autotune"


def test_target_network_updates_more_frequently_for_first_1000_updates() -> None:
    due_updates = [
        update
        for update in range(1, 3_001)
        if target_network_update_due(
            update,
            interval=1_000,
            initial_interval=200,
            initial_steps=1_000,
        )
    ]

    assert due_updates == [200, 400, 600, 800, 1_000, 2_000, 3_000]


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


def test_visit_temperature_uses_efficientzero_v1_collection_schedule() -> None:
    collection_steps = 100_000

    assert visit_softmax_temperature(0, collection_steps) == 1.0
    assert visit_softmax_temperature(49_999, collection_steps) == 1.0
    assert visit_softmax_temperature(50_000, collection_steps) == 0.5
    assert visit_softmax_temperature(74_999, collection_steps) == 0.5
    assert visit_softmax_temperature(75_000, collection_steps) == 0.25
    assert visit_softmax_temperature(100_000, collection_steps) == 0.25


def test_visit_temperature_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        visit_softmax_temperature(0, 0)
    with pytest.raises(ValueError, match="trained_steps"):
        visit_softmax_temperature(-1, 100)
