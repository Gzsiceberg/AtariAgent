import pytest

from atariagent.training import (
    representative_checkpoint_path,
    representative_checkpoint_updates,
)


def test_representative_checkpoints_cover_complete_training() -> None:
    assert representative_checkpoint_updates(120_000, 10) == (
        12_000,
        24_000,
        36_000,
        48_000,
        60_000,
        72_000,
        84_000,
        96_000,
        108_000,
        120_000,
    )


def test_representative_checkpoints_handle_short_training() -> None:
    assert representative_checkpoint_updates(3, 10) == (1, 2, 3)
    with pytest.raises(ValueError, match="total_updates"):
        representative_checkpoint_updates(0)


def test_representative_checkpoint_path_is_numbered_next_to_latest() -> None:
    assert representative_checkpoint_path(
        "checkpoints/agent_latest.pt", 12_000
    ).as_posix() == "checkpoints/agent_update_00012000.pt"
