"""Batched EfficientZero agent for selecting actions with MCTS."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import random
from typing import TypeAlias

from einops import rearrange
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from .models import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from .search import MCTS, MCTSConfig, SearchResult


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
    # Stack uint8 observations before conversion instead of allocating and
    # normalizing one float tensor per environment.
    frames = torch.from_numpy(
        np.stack(tuple(np.asarray(observation) for observation in observations))
    ).float()
    frames.div_(255.0)
    if frames.ndim == 5:
        return rearrange(
            frames,
            "batch stack height width channels -> "
            "batch (stack channels) height width",
        )
    if frames.ndim == 4:
        return frames
    raise ValueError(
        "Atari observations must have shape (stack, H, W, C) or (channels, H, W)"
    )


@lru_cache(maxsize=32)
def _categorical_support(
    support_min: int,
    support_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Cache the small immutable support used by repeated scalar decoding."""
    return torch.arange(
        support_min,
        support_max + 1,
        device=device,
        dtype=dtype,
    )


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
    support = _categorical_support(
        support_min,
        support_max,
        logits.device,
        logits.dtype,
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


class BatchedNetworkEvaluator:
    """Evaluate MCTS leaves with batched dynamics and prediction networks."""

    def __init__(
        self,
        dynamics_network: nn.Module,
        prediction_network: nn.Module,
        *,
        action_space_size: int,
        value_decoder: ScalarDecoder,
        value_prefix_decoder: ScalarDecoder,
    ) -> None:
        self.dynamics_network = dynamics_network
        self.prediction_network = prediction_network
        self.action_space_size = action_space_size
        self.value_decoder = value_decoder
        self.value_prefix_decoder = value_prefix_decoder
        reward_prediction = getattr(dynamics_network, "reward_prediction", None)
        hidden_size = getattr(reward_prediction, "hidden_size", None)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError(
                "dynamics_network.reward_prediction.hidden_size "
                "must be an integer"
            )
        if hidden_size <= 0:
            raise ValueError("recurrent hidden size must be positive")
        self.hidden_size = hidden_size

    @torch.inference_mode()
    def evaluate_tensors(
        self,
        states: Tensor,
        actions: Tensor,
        value_prefix_hidden: RewardHidden,
        reset_value_prefix: Tensor | None = None,
    ) -> tuple[Tensor, RewardHidden, Tensor, Tensor, Tensor]:
        """Evaluate packed MCTS leaves without per-root Python objects."""
        batch_size = states.shape[0]
        if actions.shape != (batch_size, 1):
            raise ValueError("actions must have shape (batch_size, 1)")
        hidden, cell = value_prefix_hidden
        if reset_value_prefix is not None:
            if reset_value_prefix.shape != (batch_size,):
                raise ValueError("reset mask must have shape (batch_size,)")
            reset = reset_value_prefix.reshape(1, batch_size, 1)
            hidden = hidden.masked_fill(reset, 0.0)
            cell = cell.masked_fill(reset, 0.0)

        next_states, next_hidden, value_prefix_logits = self.dynamics_network(
            states,
            actions,
            (hidden, cell),
        )
        policy_logits, value_logits = self.prediction_network(next_states)
        self.validate_policy(policy_logits, batch_size)
        values = self.decode(self.value_decoder, value_logits, "value_decoder")
        value_prefixes = self.decode(
            self.value_prefix_decoder,
            value_prefix_logits,
            "value_prefix_decoder",
        )
        return next_states, next_hidden, value_prefixes, values, policy_logits

    def validate_policy(self, policy_logits: Tensor, batch_size: int) -> None:
        expected = (batch_size, self.action_space_size)
        if tuple(policy_logits.shape) != expected:
            raise ValueError(
                f"prediction network policy shape must be {expected}, "
                f"got {tuple(policy_logits.shape)}"
            )

    @staticmethod
    def decode(decoder: ScalarDecoder, logits: Tensor, name: str) -> Tensor:
        values = decoder(logits)
        if values.numel() != logits.shape[0]:
            raise ValueError(f"{name} must return one scalar per batch item")
        return values.reshape(logits.shape[0])

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RewardHidden:
        """Allocate one packed zero recurrent state for native MCTS."""
        hidden = torch.zeros(
            1, batch_size, self.hidden_size, device=device, dtype=dtype
        )
        return hidden, torch.zeros_like(hidden)


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
        mcts_config: MCTSConfig | None = None,
        mcts_rng: random.Random | None = None,
        value_decoder: ScalarDecoder = categorical_to_scalar,
        value_prefix_decoder: ScalarDecoder = categorical_to_scalar,
    ) -> None:
        super().__init__()
        if action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
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
        self.value_decoder = value_decoder
        self.value_prefix_decoder = value_prefix_decoder
        self.recurrent_evaluator = BatchedNetworkEvaluator(
            self.dynamics_network,
            self.prediction_network,
            action_space_size=action_space_size,
            value_decoder=self.value_decoder,
            value_prefix_decoder=self.value_prefix_decoder,
        )
        self.mcts = MCTS(
            mcts_config,
            evaluator=self.recurrent_evaluator,
            rng=mcts_rng,
        )

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
        if was_training:
            self.eval()
        try:
            states = self.representation_network(observations)
            policy_logits, value_logits = self.prediction_network(states)
            self.recurrent_evaluator.validate_policy(
                policy_logits, states.shape[0]
            )
            values = self.recurrent_evaluator.decode(
                self.recurrent_evaluator.value_decoder,
                value_logits,
                "value_decoder",
            )

            search_batch = self.mcts.search_batch(
                states,
                values,
                policy_logits,
                add_exploration_noise=add_exploration_noise,
            )
            search_results = self.mcts.materialize_results(
                search_batch,
                temperature=temperature,
            )
            return AgentOutput(
                actions=tuple(result.action for result in search_results),
                search_results=search_results,
            )
        finally:
            if was_training:
                self.train()

    def _prepare_observations(
        self, observations: Tensor | Sequence[AtariObservation]
    ) -> Tensor:
        if isinstance(observations, Tensor):
            batch = observations
            if batch.ndim == 3:
                batch = rearrange(
                    batch, "channels height width -> 1 channels height width"
                )
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

Agent = AtariAgent


__all__ = [
    "Agent",
    "AgentOutput",
    "AtariAgent",
    "AtariObservation",
    "BatchedNetworkEvaluator",
    "ScalarDecoder",
    "atari_observation_tensor",
    "batch_atari_observations",
    "categorical_to_scalar",
]
