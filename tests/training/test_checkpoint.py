import pytest

from atariagent.training import (
    representative_checkpoint_path,
    representative_checkpoint_updates,
)


def test_representative_checkpoints_use_phase_specific_intervals() -> None:
    assert representative_checkpoint_updates(100_000, 20_000) == (
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
        60_000,
        70_000,
        80_000,
        90_000,
        100_000,
        105_000,
        110_000,
        115_000,
        120_000,
    )


def test_representative_checkpoints_include_each_phase_endpoint() -> None:
    assert representative_checkpoint_updates(
        12, 7, collection_interval=5, final_interval=3
    ) == (5, 10, 12, 15, 18, 19)
    assert representative_checkpoint_updates(3, 0) == (3,)


def test_representative_checkpoints_validate_schedule() -> None:
    with pytest.raises(ValueError, match="collection_updates"):
        representative_checkpoint_updates(0, 1)
    with pytest.raises(ValueError, match="final_updates"):
        representative_checkpoint_updates(1, -1)
    with pytest.raises(ValueError, match="collection_interval"):
        representative_checkpoint_updates(1, 1, collection_interval=0)
    with pytest.raises(ValueError, match="final_interval"):
        representative_checkpoint_updates(1, 1, final_interval=0)


def test_representative_checkpoint_path_is_numbered_next_to_latest() -> None:
    assert representative_checkpoint_path(
        "checkpoints/agent_latest.pt", 12_000
    ).as_posix() == "checkpoints/agent_update_00012000.pt"
