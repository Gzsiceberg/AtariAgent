"""Learning-signal diagnostics must be detached, masked, and compile-safe."""

import pytest
import torch

from atariagent.training.learner import _LearnerUnroll


@pytest.mark.parametrize("values", [[1.0], [1.0] * 8, [0.01, 0.05, 0.1, 1.0], [0.0] * 4])
def test_importance_weight_distribution(values):
    weights = torch.tensor(values, requires_grad=True)
    actual = _LearnerUnroll._importance_statistics(weights)
    expected = torch.stack((
        weights.mean(),
        weights.sum().square()
        / (weights.square().sum() * weights.numel()).clamp_min(torch.finfo(torch.float32).tiny),
    ))
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_feature_variance_excludes_padding_and_uses_batch_axis(dtype):
    # Each coordinate varies by 2 between the valid samples: variance is 1.
    state = torch.tensor([[0., 10.], [2., 12.], [1000., -1000.]], dtype=dtype)
    state = state.reshape(3, 1, 1, 2).requires_grad_()
    actual = _LearnerUnroll._feature_variance(state, torch.tensor([True, True, False]))
    assert actual.item() == pytest.approx(1.0)
    assert actual.dtype == torch.float32
    assert not actual.requires_grad

    # Spatial structure alone must not conceal observation-independent collapse.
    constant = state[:1].expand(3, -1, -1, -1)
    assert _LearnerUnroll._feature_variance(constant, torch.ones(3, dtype=torch.bool)) == 0


@pytest.mark.parametrize("mask", [[False, False], [True, False]])
def test_feature_variance_empty_or_singleton_is_zero(mask):
    state = torch.randn(2, 4, 3, 3)
    actual = _LearnerUnroll._feature_variance(state, torch.tensor(mask))
    assert actual == 0
    assert torch.isfinite(actual)


def test_learning_signal_diagnostics_support_fullgraph_compilation():
    def diagnostics(state, mask, weights):
        return (
            _LearnerUnroll._feature_variance(state, mask),
            _LearnerUnroll._importance_statistics(weights),
        )

    compiled = torch.compile(diagnostics, backend="eager", fullgraph=True)
    state = torch.randn(4, 2, 3, 3, requires_grad=True)
    mask = torch.tensor([True, False, True, True])
    weights = torch.tensor([0.01, 0.1, 0.5, 1.0])
    actual = compiled(state, mask, weights)
    expected = diagnostics(state, mask, weights)
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference)
        assert not output.requires_grad
