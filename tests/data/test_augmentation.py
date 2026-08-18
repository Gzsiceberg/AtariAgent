import pytest
import torch

from atariagent.data import Intensity, RandomShiftsAug, Transforms


def test_shift_preserves_shape_and_alignment_between_stacked_channels() -> None:
    base = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    images = torch.cat((base, base + 100.0, base + 200.0), dim=1)

    shifted = RandomShiftsAug(pad=2)(images)

    assert shifted.shape == images.shape
    torch.testing.assert_close(
        shifted[:, 1] - shifted[:, 0], torch.full((1, 8, 8), 100.0)
    )
    torch.testing.assert_close(
        shifted[:, 2] - shifted[:, 0], torch.full((1, 8, 8), 200.0)
    )


def test_shift_uses_replication_padding() -> None:
    torch.manual_seed(0)
    images = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    shifted = RandomShiftsAug(pad=4)(images)

    assert shifted.shape == images.shape
    assert set(shifted.flatten().tolist()) <= {1.0, 2.0, 3.0, 4.0}


def test_intensity_uses_one_clipped_multiplier_per_batch_item() -> None:
    torch.manual_seed(3)
    images = torch.ones(4, 3, 5, 5)

    augmented = Intensity(scale=0.05)(images)

    multipliers = augmented[:, :1, :1, :1]
    torch.testing.assert_close(augmented, multipliers.expand_as(augmented))
    assert torch.all(multipliers >= 0.9)
    assert torch.all(multipliers <= 1.1)


def test_transforms_restore_leading_dimensions_without_tracking_gradients() -> None:
    images = torch.ones(2, 3, 4, 6, 6, requires_grad=True)

    augmented = Transforms(
        ["shift", "intensity"], image_shape=(6, 6)
    )(images)

    assert augmented.shape == images.shape
    assert not augmented.requires_grad


def test_transforms_reject_unknown_augmentation() -> None:
    with pytest.raises(ValueError, match="unsupported augmentation"):
        Transforms(["rotate"])
