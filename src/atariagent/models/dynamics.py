"""Recurrent dynamics and value-prefix prediction for EfficientZero."""

from einops import rearrange, repeat
from jaxtyping import Float, Int
import torch
from torch import Tensor, nn

from atariagent.typecheck import runtime_typed

from .representation import ResidualBlock, conv3x3


LSTMHidden = tuple[
    Float[Tensor, "1 batch 512"],
    Float[Tensor, "1 batch 512"],
]


class RewardPredictionNetwork(nn.Module):
    """Predict categorical value-prefix logits from a latent state.

    The recurrent state accumulates reward information across model unrolls.
    The final linear layer is zero-initialized so the initial categorical
    prediction is stable and uniform after softmax.
    """

    hidden_size = 512
    support_size = 601

    def __init__(self, *, batch_norm_momentum: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_size=16 * 6 * 6, hidden_size=self.hidden_size)
        self.lstm_output = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.BatchNorm1d(32, momentum=batch_norm_momentum),
            nn.ELU(inplace=True),
            nn.Linear(32, self.support_size),
        )
        output_layer = self.projection[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    @staticmethod
    def initial_hidden(
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LSTMHidden:
        """Create the zero-valued LSTM state used at the root of an unroll."""
        shape = (1, batch_size, RewardPredictionNetwork.hidden_size)
        hidden = torch.zeros(shape, device=device, dtype=dtype)
        cell = torch.zeros(shape, device=device, dtype=dtype)
        return hidden, cell

    @runtime_typed
    def forward(
        self,
        state: Float[Tensor, "batch 64 6 6"],
        hidden: LSTMHidden | None = None,
    ) -> tuple[Float[Tensor, "batch 601"], LSTMHidden]:
        if hidden is None:
            hidden = self.initial_hidden(
                state.shape[0], device=state.device, dtype=state.dtype
            )

        features = rearrange(
            self.features(state),
            "batch channels height width -> 1 batch (channels height width)",
        )
        recurrent_output, next_hidden = self.lstm(features, hidden)
        recurrent_output = self.lstm_output(
            rearrange(recurrent_output, "1 batch hidden -> batch hidden")
        )
        value_prefix = self.projection(recurrent_output)
        return value_prefix, next_hidden


class DynamicsNetwork(nn.Module):
    """Predict the next latent state and value prefix for a discrete action.

    A normalized action index is broadcast over one plane, projected to 16
    channels, layer-normalized, and concatenated with the 64-plane latent
    state. The dynamics convolution uses stride one; this keeps its output
    compatible with the historical-state residual link.

    ``forward`` returns ``(next_state, next_hidden, value_prefix)`` to match the
    recurrent dynamics contract used by the EfficientZero reference model.
    """

    def __init__(
        self,
        action_space_size: int,
        *,
        batch_norm_momentum: float = 0.1,
        scale_state_gradient: bool = True,
        action_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
        if not isinstance(scale_state_gradient, bool):
            raise TypeError("scale_state_gradient must be a boolean")
        if action_embedding_dim <= 0:
            raise ValueError("action_embedding_dim must be positive")

        self.action_space_size = action_space_size
        self.scale_state_gradient = scale_state_gradient
        self.action_embedding_dim = action_embedding_dim
        self.action_projection = nn.Conv2d(
            1, action_embedding_dim, kernel_size=1
        )
        self.action_normalization = nn.LayerNorm(
            [action_embedding_dim, 6, 6]
        )
        self.transition = nn.Sequential(
            conv3x3(64 + action_embedding_dim, 64),
            nn.BatchNorm2d(64, momentum=batch_norm_momentum),
        )
        self.relu = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(64, batch_norm_momentum=batch_norm_momentum)
        self.reward_prediction = RewardPredictionNetwork(
            batch_norm_momentum=batch_norm_momentum
        )

    @runtime_typed
    def forward(
        self,
        state: Float[Tensor, "batch 64 6 6"],
        action: Int[Tensor, "batch 1"],
        reward_hidden: LSTMHidden | None = None,
    ) -> tuple[
        Float[Tensor, "batch 64 6 6"],
        LSTMHidden,
        Float[Tensor, "batch 601"],
    ]:
        action_plane = repeat(
            action, "batch 1 -> batch 1 height width", height=6, width=6
        ).to(dtype=state.dtype)
        action_plane = action_plane / self.action_space_size
        action_plane = self.relu(
            self.action_normalization(self.action_projection(action_plane))
        )

        transition = self.transition(torch.cat((state, action_plane), dim=1))
        next_state = self.residual(self.relu(transition + state))
        if self.scale_state_gradient:
            # Placing this forward identity before reward prediction scales
            # every gradient through the recurrent state without tensor hooks.
            next_state = next_state * 0.5 + next_state.detach() * 0.5
        value_prefix, next_hidden = self.reward_prediction(next_state, reward_hidden)
        return next_state, next_hidden, value_prefix
