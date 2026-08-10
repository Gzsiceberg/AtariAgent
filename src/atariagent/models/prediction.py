"""Policy and categorical value prediction for EfficientZero."""

from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from .representation import ResidualBlock


class _PredictionHead(nn.Module):
    """Common convolutional MLP head used for policy and value prediction."""

    def __init__(
        self,
        output_size: int,
        *,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if output_size <= 0:
            raise ValueError("output_size must be positive")

        self.output_size = output_size
        self.features = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(16 * 6 * 6, 32),
            nn.BatchNorm1d(32, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_size),
        )

        output_layer = self.projection[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, state: Float[Tensor, "batch 64 6 6"]
    ) -> Float[Tensor, "batch output"]:
        features = rearrange(
            self.features(state),
            "batch channels height width -> batch (channels height width)",
        )
        return self.projection(features)


class PolicyNetwork(_PredictionHead):
    """Predict one unnormalized policy logit for each discrete action."""

    def __init__(
        self,
        action_space_size: int,
        *,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        if action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
        self.action_space_size = action_space_size
        super().__init__(
            action_space_size,
            batch_norm_momentum=batch_norm_momentum,
        )


class ValueNetwork(_PredictionHead):
    """Predict logits over the categorical scalar-value support."""

    support_size = 601

    def __init__(
        self,
        *,
        support_size: int = support_size,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        self.support_size = support_size
        super().__init__(support_size, batch_norm_momentum=batch_norm_momentum)


class PredictionNetwork(nn.Module):
    """Predict policy and value from a 64x6x6 latent state.

    A residual block is shared before the two prediction heads, matching the
    Atari architecture in the EfficientZero reference implementation. The
    return order is ``(policy_logits, value_logits)``.
    """

    def __init__(
        self,
        action_space_size: int,
        *,
        value_support_size: int = ValueNetwork.support_size,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.residual = ResidualBlock(
            64, batch_norm_momentum=batch_norm_momentum
        )
        self.policy = PolicyNetwork(
            action_space_size,
            batch_norm_momentum=batch_norm_momentum,
        )
        self.value = ValueNetwork(
            support_size=value_support_size,
            batch_norm_momentum=batch_norm_momentum,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, state: Float[Tensor, "batch 64 6 6"]
    ) -> tuple[
        Float[Tensor, "batch actions"],
        Float[Tensor, "batch support"],
    ]:
        prediction_state = self.residual(state)
        return self.policy(prediction_state), self.value(prediction_state)


__all__ = ["PolicyNetwork", "PredictionNetwork", "ValueNetwork"]
