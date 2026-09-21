import io
from pathlib import Path

from hydra import compose, initialize_config_dir

import pytest
import torch

from atariagent.training.config import ReanalysisConfig, register_train_agent_config
from atariagent.training.target_delay import TargetDelay


def test_delay_configuration() -> None:
    assert ReanalysisConfig().target_update_delay == 0
    register_train_agent_config()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        default = compose(config_name="train_agent")
        delayed = compose(
            config_name="train_agent", overrides=["reanalysis.target_update_delay=200"]
        )
    assert default.reanalysis.target_update_delay == 0
    assert delayed.reanalysis.target_update_delay == 200


def weights(step: int) -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor(step)}


@pytest.mark.parametrize("interval,delay", [(200, 0), (200, 200), (200, 73), (200, 450), (1000, 200)])
def test_publishes_exact_delayed_snapshot(interval: int, delay: int) -> None:
    history = TargetDelay(interval, delay, weights(0))
    captures = []
    for step in range(1, 3001):
        def capture():
            captures.append(step)
            return weights(step)

        state = history.advance(step, capture)
        if step % interval:
            assert state is None
        else:
            assert state is not None
            assert state["weight"].item() == max(0, step - delay)
        assert len(history.pending) <= delay // interval + 3
    assert all((step + delay) % interval == 0 for step in captures)


def test_previous_snapshot_schedule() -> None:
    history = TargetDelay(200, 200, weights(0))
    published = {}
    for step in range(1, 601):
        state = history.advance(step, lambda: weights(step))
        if state is not None:
            published[step] = state["weight"].item()
    assert published == {200: 0, 400: 200, 600: 400}


@pytest.mark.parametrize("delay", [0, 73, 200, 450])
def test_resume_retains_pending_snapshots(delay: int) -> None:
    original = TargetDelay(200, delay, weights(0))
    for step in range(1, 551):
        original.advance(step, lambda: weights(step))
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    resumed = TargetDelay(200, delay, weights(-1))
    resumed.load_state_dict(torch.load(buffer, weights_only=True))
    for step in range(551, 1201):
        expected = original.advance(step, lambda: weights(step))
        actual = resumed.advance(step, lambda: weights(step))
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert torch.equal(actual["weight"], expected["weight"])


@pytest.mark.parametrize("delay,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_delay(delay, error) -> None:
    with pytest.raises(error):
        TargetDelay(200, delay, weights(0))


def test_resume_rejects_changed_schedule() -> None:
    history = TargetDelay(200, 200, weights(0))
    with pytest.raises(ValueError, match="do not match"):
        TargetDelay(200, 0, weights(0)).load_state_dict(history.state_dict())
