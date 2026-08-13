import pytest

from atariagent.training.muzero_config import (
    TrainingConfig,
    next_collection_vector_steps,
    visit_softmax_temperature,
)


def test_target_network_uses_efficientzero_hard_copy_interval() -> None:
    config = TrainingConfig()
    assert config.use_target_network_reanalysis
    assert config.policy_reanalysis_chunk_size == 768
    assert config.cache_reanalyzed_targets
    assert config.reanalysis_start_step == 1_000
    assert config.reanalysis_prefetch_batches == 2
    assert config.batch_max_in_flight == 3
    assert config.batch_ready_prefetch == 2
    assert config.batch_worker_timeout_seconds == pytest.approx(600.0)
    assert config.reanalysis_timeout_seconds == pytest.approx(600.0)
    assert config.reanalysis_max_weight_lag == 200
    assert config.reanalysis_worker_num_threads == 4
    assert config.target_update_interval == 200
    assert config.compile_mode == "max-autotune"


def test_collection_steps_stop_at_exact_transition_budget() -> None:
    collected = 0
    while collected < 100_000:
        vector_steps = next_collection_vector_steps(
            collected, 100_000, num_envs=4, max_vector_steps=100
        )
        collected += vector_steps * 4

    assert collected == 100_000
    assert next_collection_vector_steps(100_000, 100_000, 4, 100) == 0


def test_visit_temperature_uses_efficientzero_schedule() -> None:
    training_steps = 100_000

    assert visit_softmax_temperature(0, training_steps) == 1.0
    assert visit_softmax_temperature(49_999, training_steps) == 1.0
    assert visit_softmax_temperature(50_000, training_steps) == 0.5
    assert visit_softmax_temperature(74_999, training_steps) == 0.5
    assert visit_softmax_temperature(75_000, training_steps) == 0.25
    assert visit_softmax_temperature(100_000, training_steps) == 0.25


def test_visit_temperature_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="training_steps"):
        visit_softmax_temperature(0, 0)
    with pytest.raises(ValueError, match="trained_steps"):
        visit_softmax_temperature(-1, 100)
