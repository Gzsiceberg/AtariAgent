import torch
from torch import nn

from atariagent.models import RepresentationNetwork, ResidualBlock


def test_representation_output_shape() -> None:
    model = RepresentationNetwork(in_channels=4)

    output = model(torch.randn(2, 4, 96, 96))

    assert output.shape == (2, 64, 6, 6)


def test_representation_intermediate_resolutions() -> None:
    model = RepresentationNetwork((12, 96, 96))
    resolutions: list[tuple[int, int]] = []
    stages = [
        model.stem,
        model.residual_48,
        model.downsample_24,
        model.residual_24,
        model.pool_12,
        model.residual_12,
        model.pool_6,
        model.residual_6,
    ]
    hooks = [
        stage.register_forward_hook(
            lambda _module, _inputs, output: resolutions.append(output.shape[-2:])
        )
        for stage in stages
    ]

    try:
        model(torch.randn(1, 12, 96, 96))
    finally:
        for hook in hooks:
            hook.remove()

    assert resolutions == [
        (48, 48),
        (48, 48),
        (24, 24),
        (24, 24),
        (12, 12),
        (12, 12),
        (6, 6),
        (6, 6),
    ]


def test_all_spatial_operations_use_three_by_three_kernels() -> None:
    model = RepresentationNetwork(4)

    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    pools = [module for module in model.modules() if isinstance(module, nn.AvgPool2d)]

    assert convolutions
    assert pools
    assert all(module.kernel_size == (3, 3) for module in convolutions)
    assert all(module.kernel_size == 3 for module in pools)


def test_representation_supports_backward() -> None:
    model = RepresentationNetwork(4)
    observation = torch.randn(2, 4, 96, 96, requires_grad=True)

    model(observation).mean().backward()

    assert observation.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_downsample_residual_block_projects_identity() -> None:
    block = ResidualBlock(32, 64, stride=2)

    assert block(torch.randn(2, 32, 48, 48)).shape == (2, 64, 24, 24)
