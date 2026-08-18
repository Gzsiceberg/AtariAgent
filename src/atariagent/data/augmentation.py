"""EfficientZero image augmentations for Atari observations.

Spatial and intensity parameters are sampled once per batch item, so all
channels in a packed frame stack remain aligned.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional


class RandomShiftsAug(nn.Module):
    """Randomly crop an image after replication padding."""

    def __init__(self, pad: int = 4) -> None:
        super().__init__()
        if isinstance(pad, bool) or not isinstance(pad, int):
            raise TypeError("pad must be an integer")
        if pad < 0:
            raise ValueError("pad must be non-negative")
        self.pad = pad

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (batch, channels, height, width)")
        if self.pad == 0:
            return inputs

        batch_size, _, height, width = inputs.shape
        padded = functional.pad(
            inputs,
            (self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )
        offsets = torch.randint(
            0,
            2 * self.pad + 1,
            (batch_size, 2),
            device=inputs.device,
        )
        rows = offsets[:, :1] + torch.arange(height, device=inputs.device)
        columns = offsets[:, 1:] + torch.arange(width, device=inputs.device)
        batch_indices = torch.arange(batch_size, device=inputs.device)[:, None, None]

        # Index the channels-last view to select a different integer crop for
        # each batch item without a Python loop or grid-sampling interpolation.
        shifted = padded.permute(0, 2, 3, 1)[
            batch_indices,
            rows[:, :, None],
            columns[:, None, :],
        ]
        return shifted.permute(0, 3, 1, 2)


class Intensity(nn.Module):
    """Apply clipped Gaussian multiplicative noise per batch item."""

    def __init__(self, scale: float = 0.05) -> None:
        super().__init__()
        if scale < 0.0:
            raise ValueError("scale must be non-negative")
        self.scale = scale

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (batch, channels, height, width)")
        noise = torch.randn(
            (inputs.shape[0], 1, 1, 1),
            device=inputs.device,
            dtype=inputs.dtype,
        ).clamp(-2.0, 2.0)
        return inputs * (1.0 + self.scale * noise)


class Transforms(nn.Module):
    """Compose the Atari augmentations used by EfficientZero.

    Inputs may have leading dimensions before ``(channels, height, width)``.
    Those dimensions are flattened while transforms are applied and restored
    in the result.
    """

    _SUPPORTED = frozenset({"shift", "intensity", "none"})

    def __init__(
        self,
        augmentation: Sequence[str],
        shift_delta: int = 4,
        image_shape: tuple[int, int] = (96, 96),
        intensity_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if len(image_shape) != 2 or any(size <= 0 for size in image_shape):
            raise ValueError("image_shape must contain two positive dimensions")

        self.augmentation = tuple(augmentation)
        self.image_shape = image_shape
        transforms: list[nn.Module] = []
        for name in self.augmentation:
            if name == "shift":
                transformation = RandomShiftsAug(shift_delta)
            elif name == "intensity":
                transformation = Intensity(intensity_scale)
            elif name == "none":
                transformation = nn.Identity()
            else:
                supported = ", ".join(sorted(self._SUPPORTED))
                raise ValueError(
                    f"unsupported augmentation {name!r}; expected one of {supported}"
                )
            transforms.append(transformation)
        self.transforms = nn.ModuleList(transforms)

    @staticmethod
    def apply_transforms(
        transforms: Sequence[nn.Module], images: Tensor
    ) -> Tensor:
        """Apply transforms in their configured order."""
        for transformation in transforms:
            images = transformation(images)
        return images

    @torch.no_grad()
    def transform(self, images: Tensor) -> Tensor:
        """Augment images while preserving all leading dimensions."""
        if images.ndim < 4:
            raise ValueError("images must have at least four dimensions")
        if images.shape[-2:] != self.image_shape:
            raise ValueError(
                f"expected image shape {self.image_shape}, got {images.shape[-2:]}"
            )
        flat_images = images.reshape(-1, *images.shape[-3:])
        flat_images = self.apply_transforms(self.transforms, flat_images)
        return flat_images.reshape(*images.shape[:-3], *flat_images.shape[1:])

    def forward(self, images: Tensor) -> Tensor:
        return self.transform(images)


__all__ = ["Intensity", "RandomShiftsAug", "Transforms"]
