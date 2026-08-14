import pytest
import torch

from atariagent.data import Intensity, Transforms


def test_shift_preserves_shape_and_alignment_between_stacked_channels() -> None:
    base = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    images = torch.cat((base, base + 100.0, base + 200.0), dim=1)
    transform = Transforms(["shift"], shift_delta=2, image_shape=(8, 8))

    shifted = transform(images)

    assert shifted.shape == images.shape
    torch.testing.assert_close(
        shifted[:, 1] - shifted[:, 0], torch.full((1, 8, 8), 100.0)
    )
    torch.testing.assert_close(
        shifted[:, 2] - shifted[:, 0], torch.full((1, 8, 8), 200.0)
    )


def test_intensity_uses_one_bounded_multiplier_per_batch_item() -> None:
    torch.manual_seed(3)
    images = torch.ones(4, 3, 5, 5, requires_grad=True)

    augmented = Intensity(scale=0.05)(images)

    multipliers = augmented[:, :1, :1, :1]
    torch.testing.assert_close(augmented, multipliers.expand_as(augmented))
    assert torch.all(multipliers >= 0.9)
    assert torch.all(multipliers <= 1.1)


def test_transforms_restore_leading_dimensions_without_tracking_gradients() -> None:
    images = torch.ones(2, 3, 4, 6, 6, requires_grad=True)

    augmented = Transforms(["intensity"])(images)

    assert augmented.shape == images.shape
    assert not augmented.requires_grad


def test_transforms_reject_unknown_augmentation() -> None:
    with pytest.raises(ValueError, match="unsupported augmentation"):
        Transforms(["rotate"])
