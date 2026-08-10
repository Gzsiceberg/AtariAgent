import pytest

from atariagent.training.muzero_config import (
    linear_priority_beta,
    visit_softmax_temperature,
)


def test_priority_beta_uses_efficientzero_schedule() -> None:
    assert linear_priority_beta(0, 120_000, 0.4, 1.0) == pytest.approx(0.4)
    assert linear_priority_beta(60_000, 120_000, 0.4, 1.0) == pytest.approx(0.7)
    assert linear_priority_beta(120_000, 120_000, 0.4, 1.0) == pytest.approx(1.0)
    assert linear_priority_beta(130_000, 120_000, 0.4, 1.0) == pytest.approx(1.0)


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
