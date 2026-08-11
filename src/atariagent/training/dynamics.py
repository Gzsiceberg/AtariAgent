"""Training utilities for recurrent latent dynamics."""

from dataclasses import dataclass

from einops import rearrange
from jaxtyping import Float, Int
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from atariagent.models.consistency import ConsistencyNetwork, consist_loss_func
from atariagent.models.dynamics import DynamicsNetwork, LSTMHidden
from atariagent.models.representation import RepresentationNetwork
from atariagent.typecheck import runtime_typed


@runtime_typed
def scalar_reward_loss(
    logits: Float[Tensor, "batch support"],
    target: Float[Tensor, "batch"],
    *,
    support_min: int = -300,
    support_max: int = 300,
) -> Float[Tensor, "batch"]:
    """Cross-entropy against a scalar projected onto adjacent support atoms."""
    expected_size = support_max - support_min + 1
    if logits.shape[1] != expected_size:
        raise ValueError(
            f"expected {expected_size} reward logits, got {logits.shape[1]}"
        )

    target = target.clamp(support_min, support_max) - support_min
    lower = target.floor().long()
    upper = target.ceil().long()
    upper_weight = target - lower
    lower_weight = 1.0 - upper_weight

    target_distribution = torch.zeros_like(logits)
    lower = rearrange(lower, "batch -> batch 1")
    upper = rearrange(upper, "batch -> batch 1")
    lower_weight = rearrange(lower_weight, "batch -> batch 1")
    upper_weight = rearrange(upper_weight, "batch -> batch 1")
    target_distribution.scatter_add_(1, lower, lower_weight)
    target_distribution.scatter_add_(1, upper, upper_weight)
    return -(target_distribution * functional.log_softmax(logits, dim=-1)).sum(dim=-1)


@dataclass(frozen=True)
class DynamicsTrainMetrics:
    loss: float
    reward_loss: float
    consistency_loss: float


class DynamicsTrainer:
    """Optimize representation, dynamics, and consistency networks together."""

    def __init__(
        self,
        representation: RepresentationNetwork,
        dynamics: DynamicsNetwork,
        consistency: ConsistencyNetwork,
        *,
        learning_rate: float = 1e-3,
        unroll_steps: int = 5,
        lstm_horizon: int = 5,
        consistency_weight: float = 2.0,
    ) -> None:
        if unroll_steps <= 0:
            raise ValueError("unroll_steps must be positive")
        if lstm_horizon <= 0:
            raise ValueError("lstm_horizon must be positive")

        self.representation = representation
        self.dynamics = dynamics
        self.consistency = consistency
        self.unroll_steps = unroll_steps
        self.lstm_horizon = lstm_horizon
        self.consistency_weight = consistency_weight
        parameters = list(representation.parameters())
        parameters += list(dynamics.parameters())
        parameters += list(consistency.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def _reset_hidden(self, state: Tensor) -> LSTMHidden:
        return self.dynamics.reward_prediction.initial_hidden(
            state.shape[0], device=state.device, dtype=state.dtype
        )

    @runtime_typed
    def train_step(
        self,
        observations: Float[Tensor, "batch sequence channels 96 96"],
        actions: Int[Tensor, "batch steps 1"],
        rewards: Float[Tensor, "batch steps"],
    ) -> DynamicsTrainMetrics:
        """Train on one batch containing an initial state and its unroll targets."""
        if observations.shape[1] != self.unroll_steps + 1:
            raise ValueError("observations must contain unroll_steps + 1 frames")
        if actions.shape[1] != self.unroll_steps:
            raise ValueError("actions must contain unroll_steps entries")
        if rewards.shape[1] != self.unroll_steps:
            raise ValueError("rewards must contain unroll_steps entries")

        self.representation.train()
        self.dynamics.train()
        self.consistency.train()
        self.optimizer.zero_grad(set_to_none=True)

        state = self.representation(observations[:, 0])
        hidden: LSTMHidden | None = None
        value_prefix_target = torch.zeros(
            observations.shape[0], device=observations.device, dtype=observations.dtype
        )
        reward_losses: list[Tensor] = []
        consistency_losses: list[Tensor] = []

        for step in range(self.unroll_steps):
            state, hidden, value_prefix = self.dynamics(
                state, actions[:, step], hidden
            )
            with torch.no_grad():
                target_state = self.representation(observations[:, step + 1])
            prediction, target = self.consistency(state, target_state)

            value_prefix_target = value_prefix_target + rewards[:, step]
            reward_losses.append(
                scalar_reward_loss(value_prefix, value_prefix_target).mean()
            )
            consistency_losses.append(consist_loss_func(prediction, target).mean())

            if (step + 1) % self.lstm_horizon == 0:
                hidden = self._reset_hidden(state)
                value_prefix_target = torch.zeros_like(value_prefix_target)

        reward_loss = torch.stack(reward_losses).mean()
        consistency_loss = torch.stack(consistency_losses).mean()
        loss = reward_loss + self.consistency_weight * consistency_loss
        loss.backward()
        self.optimizer.step()

        return DynamicsTrainMetrics(
            loss=loss.detach().item(),
            reward_loss=reward_loss.detach().item(),
            consistency_loss=consistency_loss.detach().item(),
        )
