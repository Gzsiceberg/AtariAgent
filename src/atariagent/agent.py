"""Batched EfficientZero agent for selecting actions with MCTS."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from einops import rearrange
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from .models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from .search import Evaluation, MCTS, MCTSConfig, SearchResult


AtariObservation: TypeAlias = NDArray[np.uint8]
ScalarDecoder: TypeAlias = Callable[[Tensor], Tensor]
RewardHidden: TypeAlias = tuple[Tensor, Tensor]


def atari_observation_tensor(observation: AtariObservation) -> Tensor:
    """Convert one stacked Atari observation to channel-first float format."""
    frames = torch.as_tensor(np.asarray(observation)).float() / 255.0
    if frames.ndim == 4:
        return rearrange(
            frames,
            "stack height width channels -> (stack channels) height width",
        )
    if frames.ndim == 3:
        return frames
    raise ValueError(
        "Atari observations must have shape (stack, H, W, C) or (channels, H, W)"
    )


def batch_atari_observations(
    observations: Sequence[AtariObservation],
) -> Tensor:
    """Convert raw Atari observations into a channel-first float batch."""
    if not observations:
        raise ValueError("observations must not be empty")
    return torch.stack(tuple(atari_observation_tensor(obs) for obs in observations))


@torch.no_grad()
def categorical_to_scalar(
    logits: Tensor,
    *,
    support_min: int = -300,
    support_max: int = 300,
    epsilon: float = 0.001,
) -> Tensor:
    """Decode categorical logits with EfficientZero's inverse transform.

    The final dimension represents every integer in the inclusive interval
    ``[support_min, support_max]``. The returned tensor has that final
    dimension removed.
    """
    if support_min >= support_max:
        raise ValueError("support_min must be less than support_max")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    support_size = support_max - support_min + 1
    if logits.ndim == 0 or logits.shape[-1] != support_size:
        actual = logits.shape[-1] if logits.ndim else 0
        raise ValueError(
            f"expected {support_size} support logits, got {actual}"
        )

    probabilities = torch.softmax(logits, dim=-1)
    support = torch.arange(
        support_min,
        support_max + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    transformed = (probabilities * support).sum(dim=-1)

    magnitude = (
        (
            torch.sqrt(
                1
                + 4
                * epsilon
                * (transformed.abs() + 1 + epsilon)
            )
            - 1
        )
        / (2 * epsilon)
    ).square() - 1
    scalar = transformed.sign() * magnitude
    scalar = torch.nan_to_num(scalar)
    return torch.where(scalar.abs() < epsilon, 0.0, scalar)


@dataclass(frozen=True, slots=True)
class AgentOutput:
    """Actions and complete tree-search statistics for an observation batch."""

    actions: tuple[int, ...]
    search_results: tuple[SearchResult, ...]

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.search_results):
            raise ValueError("actions and search_results must have equal lengths")


class AtariAgent(nn.Module):
    """Use EfficientZero networks and batched MCTS to choose Atari actions.

    Root observations are encoded and predicted in one batch. Each MCTS
    simulation also evaluates one selected leaf per environment in one
    dynamics/prediction batch. Network execution is always performed with
    inference mode enabled and batch-normalization layers in evaluation mode.
    """

    def __init__(
        self,
        in_channels: int | Sequence[int],
        action_space_size: int,
        *,
        representation_network: nn.Module | None = None,
        dynamics_network: nn.Module | None = None,
        prediction_network: nn.Module | None = None,
        mcts: MCTS | None = None,
        mcts_config: MCTSConfig | None = None,
        value_decoder: ScalarDecoder = categorical_to_scalar,
        value_prefix_decoder: ScalarDecoder = categorical_to_scalar,
    ) -> None:
        super().__init__()
        if action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
        if mcts is not None and mcts_config is not None:
            raise ValueError("pass either mcts or mcts_config, not both")

        self.action_space_size = action_space_size
        self.representation_network = (
            representation_network
            if representation_network is not None
            else RepresentationNetwork(in_channels)
        )
        self.dynamics_network = (
            dynamics_network
            if dynamics_network is not None
            else DynamicsNetwork(action_space_size)
        )
        self.prediction_network = (
            prediction_network
            if prediction_network is not None
            else PredictionNetwork(action_space_size)
        )
        self.mcts = mcts if mcts is not None else MCTS(mcts_config)
        self.value_decoder = value_decoder
        self.value_prefix_decoder = value_prefix_decoder

    def forward(
        self,
        observations: Tensor | Sequence[AtariObservation],
        *,
        add_exploration_noise: bool = False,
        temperature: float = 0.0,
    ) -> AgentOutput:
        """Return one MCTS-selected action and search result per observation."""
        return self.act(
            observations,
            add_exploration_noise=add_exploration_noise,
            temperature=temperature,
        )

    @torch.inference_mode()
    def act(
        self,
        observations: Tensor | Sequence[AtariObservation],
        *,
        add_exploration_noise: bool = False,
        temperature: float = 0.0,
    ) -> AgentOutput:
        """Evaluate raw Atari observations or a prepared tensor batch."""
        observations = self._prepare_observations(observations)

        was_training = self.training
        self.eval()
        try:
            states = self.representation_network(observations)
            policy_logits, value_logits = self.prediction_network(states)
            self._validate_policy(policy_logits, states.shape[0])
            values = self._decode(
                self.value_decoder, value_logits, "value_decoder"
            )

            roots = tuple(
                Evaluation(
                    state=states[index],
                    value_prefix=0.0,
                    value=float(values[index]),
                    policy_logits=policy_logits[index].tolist(),
                )
                for index in range(states.shape[0])
            )
            search_results = self.mcts.search_batch(
                roots,
                self._evaluate_recurrent_batch,
                add_exploration_noise=add_exploration_noise,
                temperature=temperature,
            )
            return AgentOutput(
                actions=tuple(result.action for result in search_results),
                search_results=search_results,
            )
        finally:
            self.train(was_training)

    def _prepare_observations(
        self, observations: Tensor | Sequence[AtariObservation]
    ) -> Tensor:
        if isinstance(observations, Tensor):
            batch = observations
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
            if batch.ndim != 4:
                raise ValueError(
                    "tensor observations must have shape "
                    "(batch, channels, height, width)"
                )
            if batch.shape[0] == 0:
                raise ValueError("observations batch must not be empty")
        else:
            batch = batch_atari_observations(observations)

        parameter = next(self.parameters(), None)
        device = parameter.device if parameter is not None else batch.device
        return batch.to(device=device, dtype=torch.float32)

    def _evaluate_recurrent_batch(
        self,
        states: Sequence[Tensor],
        actions: Sequence[int],
        hidden_states: Sequence[RewardHidden | None],
    ) -> tuple[Evaluation, ...]:
        if not states:
            return ()

        state_batch = torch.stack(tuple(states))
        action_batch = torch.as_tensor(
            actions, device=state_batch.device, dtype=torch.long
        ).reshape(-1, 1)
        hidden_batch = self._batch_hidden(
            hidden_states,
            batch_size=len(states),
            device=state_batch.device,
            dtype=state_batch.dtype,
        )

        next_states, next_hidden, value_prefix_logits = self.dynamics_network(
            state_batch, action_batch, hidden_batch
        )
        policy_logits, value_logits = self.prediction_network(next_states)
        self._validate_policy(policy_logits, len(states))
        values = self._decode(
            self.value_decoder, value_logits, "value_decoder"
        )
        value_prefixes = self._decode(
            self.value_prefix_decoder,
            value_prefix_logits,
            "value_prefix_decoder",
        )

        return tuple(
            Evaluation(
                state=next_states[index],
                value_prefix=float(value_prefixes[index]),
                value=float(values[index]),
                policy_logits=policy_logits[index].tolist(),
                value_prefix_hidden=(
                    next_hidden[0][:, index : index + 1],
                    next_hidden[1][:, index : index + 1],
                ),
            )
            for index in range(len(states))
        )

    def _batch_hidden(
        self,
        hidden_states: Sequence[RewardHidden | None],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RewardHidden:
        if len(hidden_states) != batch_size:
            raise ValueError("expected one recurrent hidden state per latent state")

        hidden_size = getattr(self.dynamics_network, "reward_prediction", None)
        hidden_size = getattr(hidden_size, "hidden_size", 512)
        zero = lambda: torch.zeros(
            1, 1, hidden_size, device=device, dtype=dtype
        )
        hidden_parts: list[Tensor] = []
        cell_parts: list[Tensor] = []
        for recurrent_state in hidden_states:
            if recurrent_state is None:
                hidden_parts.append(zero())
                cell_parts.append(zero())
            else:
                hidden, cell = recurrent_state
                hidden_parts.append(hidden.to(device=device, dtype=dtype))
                cell_parts.append(cell.to(device=device, dtype=dtype))
        return torch.cat(hidden_parts, dim=1), torch.cat(cell_parts, dim=1)

    def _validate_policy(self, policy_logits: Tensor, batch_size: int) -> None:
        expected = (batch_size, self.action_space_size)
        if tuple(policy_logits.shape) != expected:
            raise ValueError(
                f"prediction network policy shape must be {expected}, "
                f"got {tuple(policy_logits.shape)}"
            )

    @staticmethod
    def _decode(
        decoder: ScalarDecoder,
        logits: Tensor,
        name: str,
    ) -> Tensor:
        values = decoder(logits)
        if values.numel() != logits.shape[0]:
            raise ValueError(f"{name} must return one scalar per batch item")
        return values.reshape(logits.shape[0])


Agent = AtariAgent


__all__ = [
    "Agent",
    "AgentOutput",
    "AtariAgent",
    "AtariObservation",
    "ScalarDecoder",
    "atari_observation_tensor",
    "batch_atari_observations",
    "categorical_to_scalar",
]
