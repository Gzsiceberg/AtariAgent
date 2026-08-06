"""Convolutional representation network for Atari observations."""

from collections.abc import Sequence

from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Create a padding-preserving 3x3 convolution."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class ResidualBlock(nn.Module):
    """A post-activation residual block with an optional downsampling skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        stride: int = 1,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.skip = conv3x3(in_channels, out_channels, stride)
        else:
            self.skip = nn.Identity()

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels out_height out_width"]:
        identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class RepresentationNetwork(nn.Module):
    """Encode 96x96 Atari observations as 64x6x6 latent states.

    Args:
        in_channels: Number of channels in a channel-first observation, or its
            ``(channels, height, width)`` shape.
        batch_norm_momentum: Momentum used by all batch-normalization layers.

    The channel count commonly includes stacked observations (for example,
    four grayscale frames have ``in_channels=4``).
    """

    output_channels = 64
    output_resolution = (6, 6)

    def __init__(
        self,
        in_channels: int | Sequence[int],
        *,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int):
            if len(in_channels) != 3:
                raise ValueError("observation shape must be (channels, height, width)")
            channels, height, width = in_channels
            if (height, width) != (96, 96):
                raise ValueError("representation network expects 96x96 observations")
            in_channels = channels
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self.in_channels = in_channels
        self.stem = nn.Sequential(
            conv3x3(in_channels, 32, stride=2),
            nn.BatchNorm2d(32, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
        )
        self.residual_48 = ResidualBlock(
            32, batch_norm_momentum=batch_norm_momentum
        )
        self.downsample_24 = ResidualBlock(
            32, 64, stride=2, batch_norm_momentum=batch_norm_momentum
        )
        self.residual_24 = ResidualBlock(
            64, batch_norm_momentum=batch_norm_momentum
        )

        self.pool_12 = self._pool_stage(64, batch_norm_momentum)
        self.residual_12 = ResidualBlock(
            64, batch_norm_momentum=batch_norm_momentum
        )
        self.pool_6 = self._pool_stage(64, batch_norm_momentum)
        self.residual_6 = ResidualBlock(
            64, batch_norm_momentum=batch_norm_momentum
        )

    @staticmethod
    def _pool_stage(channels: int, momentum: float) -> nn.Sequential:
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, observation: Float[Tensor, "batch channels 96 96"]
    ) -> Float[Tensor, "batch 64 6 6"]:
        if observation.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {observation.shape[1]}"
            )

        x = self.stem(observation)
        x = self.residual_48(x)
        x = self.downsample_24(x)
        x = self.residual_24(x)
        x = self.pool_12(x)
        x = self.residual_12(x)
        x = self.pool_6(x)
        return self.residual_6(x)
