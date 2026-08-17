"""SimSiam-style projection networks for latent-state consistency."""

from einops import rearrange
from jaxtyping import Float
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from atariagent.typecheck import runtime_typed


class Projector(nn.Module):
    """Map flattened latent states into the consistency embedding space."""

    def __init__(
        self,
        input_dim: int = 64 * 6 * 6,
        hidden_dim: int = 1024,
        output_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    @runtime_typed
    def forward(
        self, state: Float[Tensor, "batch 64 6 6"]
    ) -> Float[Tensor, "batch embedding"]:
        flattened = rearrange(state, "batch channels height width -> batch (channels height width)")
        return self.network(flattened)


class Predictor(nn.Module):
    """Predict a target projection from a dynamics-state projection."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    @runtime_typed
    def forward(
        self, projection: Float[Tensor, "batch input_dim"]
    ) -> Float[Tensor, "batch output_dim"]:
        return self.network(projection)


class ConsistencyNetwork(nn.Module):
    """Apply the shared projector and prediction head used by SimSiam."""

    def __init__(
        self,
        *,
        projection_dim: int = 1024,
        projection_hidden_dim: int = 1024,
        prediction_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.projector = Projector(
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        )
        self.predictor = Predictor(
            input_dim=projection_dim,
            hidden_dim=prediction_hidden_dim,
            output_dim=projection_dim,
        )

    @runtime_typed
    def forward(
        self,
        predicted_state: Float[Tensor, "batch 64 6 6"],
        target_state: Float[Tensor, "batch 64 6 6"],
    ) -> tuple[
        Float[Tensor, "batch projection"],
        Float[Tensor, "batch projection"],
    ]:
        prediction = self.predictor(self.projector(predicted_state))
        with torch.no_grad():
            target = self.projector(target_state)
        return prediction, target


@runtime_typed
def consist_loss_func(
    prediction: Float[Tensor, "batch embedding"],
    target: Float[Tensor, "batch embedding"],
) -> Float[Tensor, "batch"]:
    """Return per-sample negative cosine similarity (SimSiam's L2 loss)."""
    prediction = functional.normalize(prediction, dim=-1, eps=1e-5)
    target = functional.normalize(target, dim=-1, eps=1e-5)
    return -(prediction * target).sum(dim=-1)
