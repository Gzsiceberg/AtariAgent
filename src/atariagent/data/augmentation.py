"""Image augmentations used by EfficientZero V1 Atari training.

The implementation follows EfficientZero's Atari dataset transforms. A single
set of spatial parameters is applied to all channels in an observation stack,
which keeps stacked frames spatially aligned.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from kornia.augmentation import RandomAffine, RandomCrop, RandomResizedCrop
from kornia.filters import GaussianBlur2d
from torch import Tensor, nn


class Intensity(nn.Module):
    """Multiply each batch item by independently sampled clipped noise."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        if scale < 0.0:
            raise ValueError("scale must be non-negative")
        self.scale = scale

    def forward(self, inputs: Tensor) -> Tensor:
        noise = torch.randn(
            (inputs.shape[0], 1, 1, 1),
            device=inputs.device,
        )
        multiplier = 1.0 + self.scale * noise.clamp(-2.0, 2.0)
        return inputs * multiplier


class Transforms:
    """Compose the image augmentations supported by EfficientZero V1.

    Inputs may have any number of leading dimensions, followed by channel,
    height, and width. Leading dimensions are flattened before augmentation
    and restored afterward.
    """

    _SUPPORTED = frozenset(
        {"affine", "crop", "rrc", "blur", "shift", "intensity", "none"}
    )

    def __init__(
        self,
        augmentation: Sequence[str],
        shift_delta: int = 4,
        image_shape: tuple[int, int] = (96, 96),
    ) -> None:
        if shift_delta < 0:
            raise ValueError("shift_delta must be non-negative")
        if len(image_shape) != 2 or any(size <= 0 for size in image_shape):
            raise ValueError("image_shape must contain two positive dimensions")

        self.augmentation = tuple(augmentation)
        self.transforms: list[nn.Module] = []
        for name in self.augmentation:
            if name == "affine":
                transformation = RandomAffine(
                    5,
                    (0.14, 0.14),
                    (0.9, 1.1),
                    (-5, 5),
                )
            elif name == "crop":
                transformation = RandomCrop(image_shape)
            elif name == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1.0))
            elif name == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
            elif name == "shift":
                transformation = nn.Sequential(
                    nn.ReplicationPad2d(shift_delta),
                    RandomCrop(image_shape),
                )
            elif name == "intensity":
                transformation = Intensity(scale=0.05)
            elif name == "none":
                transformation = nn.Identity()
            else:
                supported = ", ".join(sorted(self._SUPPORTED))
                raise ValueError(
                    f"unsupported augmentation {name!r}; expected one of {supported}"
                )
            self.transforms.append(transformation)

    @staticmethod
    def apply_transforms(transforms: Sequence[nn.Module], images: Tensor) -> Tensor:
        """Apply transforms in their configured order."""
        for transformation in transforms:
            images = transformation(images)
        return images

    @torch.no_grad()
    def transform(self, images: Tensor) -> Tensor:
        """Augment images while preserving all leading dimensions."""
        if images.ndim < 4:
            raise ValueError("images must have at least four dimensions")
        flat_images = images.reshape(-1, *images.shape[-3:])
        processed_images = self.apply_transforms(self.transforms, flat_images)
        return processed_images.reshape(
            *images.shape[:-3],
            *processed_images.shape[1:],
        )

    def __call__(self, images: Tensor) -> Tensor:
        return self.transform(images)


__all__ = ["Intensity", "Transforms"]
