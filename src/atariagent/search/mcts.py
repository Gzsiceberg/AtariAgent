"""Native packed-tensor Monte Carlo tree search for EfficientZero."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
import random
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from ._mcts_native import BatchTree as NativeBatchTree


PackedHidden = tuple[Tensor, Tensor]
PackedEvaluation = tuple[Tensor, PackedHidden, Tensor, Tensor, Tensor]


@runtime_checkable
class PackedEvaluator(Protocol):
    """Typed network interface bound to one packed MCTS instance."""

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PackedHidden: ...

    def evaluate_tensors(
        self,
        states: Tensor,
        actions: Tensor,
        value_prefix_hidden: PackedHidden,
        reset_value_prefix: Tensor | None = None,
    ) -> PackedEvaluation: ...

    def validate_policy(self, policy_logits: Tensor, batch_size: int) -> None: ...


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Configuration for the UCT search described by EfficientZero."""

    num_simulations: int = 50
    discount: float = 0.997
    pb_c_init: float = 1.25
    pb_c_base: float = 19652.0
    value_delta_max: float = 0.01
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    value_prefix_horizon: int = 5

    def __post_init__(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if self.pb_c_init < 0.0:
            raise ValueError("pb_c_init must be non-negative")
        if self.pb_c_base <= 0.0:
            raise ValueError("pb_c_base must be positive")
        if self.value_delta_max <= 0.0:
            raise ValueError("value_delta_max must be positive")
        if self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0.0 <= self.root_exploration_fraction <= 1.0:
            raise ValueError("root_exploration_fraction must be in [0, 1]")
        if self.value_prefix_horizon <= 0:
            raise ValueError("value_prefix_horizon must be positive")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Materialized outputs for one root, stored by self-play replay."""

    action: int
    visit_counts: tuple[int, ...]
    root_value: float


@dataclass(frozen=True, slots=True)
class SearchBatchResult:
    """Contiguous native MCTS output before per-root materialization."""

    visit_counts: NDArray[np.int32]
    root_values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.visit_counts.ndim != 2:
            raise ValueError("visit_counts must have shape (roots, actions)")
        if self.root_values.shape != (self.visit_counts.shape[0],):
            raise ValueError("root_values must have shape (roots,)")


class MCTS:
    """Run native batched UCT search with one bound network evaluator."""

    def __init__(
        self,
        config: MCTSConfig | None = None,
        *,
        evaluator: PackedEvaluator,
        rng: random.Random | None = None,
    ) -> None:
        if not isinstance(evaluator, PackedEvaluator):
            raise TypeError("evaluator must implement PackedEvaluator")
        self.config = config or MCTSConfig()
        self.evaluator = evaluator
        self.rng = rng or random.Random()

    def search_batch(
        self,
        root_states: Tensor,
        root_values: Tensor,
        root_policy_logits: Tensor,
        *,
        add_exploration_noise: bool = False,
        _deterministic_ties: bool = False,
    ) -> SearchBatchResult:
        """Search packed roots and return contiguous native arrays."""
        if root_states.ndim < 2:
            raise ValueError("root_states must contain a batch dimension")
        root_count = root_states.shape[0]
        if root_count == 0:
            return SearchBatchResult(
                visit_counts=np.empty(
                    (0, root_policy_logits.shape[-1]),
                    dtype=np.int32,
                ),
                root_values=np.empty(0, dtype=np.float32),
            )
        if root_values.shape != (root_count,):
            raise ValueError("root_values must have shape (batch_size,)")
        if root_policy_logits.ndim != 2 or root_policy_logits.shape[0] != root_count:
            raise ValueError(
                "root_policy_logits must have shape (batch_size, actions)"
            )
        if root_states.device != root_values.device or (
            root_states.device != root_policy_logits.device
        ):
            raise ValueError("root tensors must be on the same device")
        evaluator = self.evaluator

        # Pack the CPU-tree inputs so CUDA performs one device-to-host
        # transfer and synchronization instead of one for each source tensor.
        root_rows = torch.cat(
            (root_values[:, None], root_policy_logits),
            dim=1,
        ).float().cpu().numpy()
        if not np.isfinite(root_rows).all():
            raise ValueError("root values and policy logits must be finite")
        root_logits = root_rows[:, 1:].astype(np.float64)
        root_logits -= root_logits.max(axis=1, keepdims=True)
        root_priors = np.exp(root_logits)
        root_priors /= root_priors.sum(axis=1, keepdims=True)
        root_priors = np.ascontiguousarray(root_priors, dtype=np.float32)
        if add_exploration_noise:
            for priors in root_priors:
                self._add_root_noise(priors)

        tree = NativeBatchTree(
            root_priors,
            np.ascontiguousarray(root_rows[:, 0]),
            np.zeros(root_count, dtype=np.float32),
            self.config.num_simulations,
            self.config.discount,
            self.config.value_prefix_horizon,
            self.config.value_delta_max,
            self.rng.getrandbits(64),
            _deterministic_ties,
        )
        device = root_states.device
        initial_hidden = evaluator.initial_hidden(
            root_count,
            device=device,
            dtype=root_states.dtype,
        )
        root_indices = torch.arange(root_count, device=device)
        state_store = root_states.new_empty(
            (
                root_count,
                self.config.num_simulations + 1,
                *root_states.shape[1:],
            )
        )
        state_store[:, 0].copy_(root_states)
        hidden_size = initial_hidden[0].shape[-1]
        hidden_store = root_states.new_empty(
            root_count,
            self.config.num_simulations + 1,
            hidden_size,
        )
        cell_store = torch.empty_like(hidden_store)
        hidden_store[:, 0].copy_(initial_hidden[0][0])
        cell_store[:, 0].copy_(initial_hidden[1][0])

        with _evaluator_inference(evaluator):
            for simulation in range(self.config.num_simulations):
                slots_array, actions_array, resets_array = tree.traverse_arrays(
                    self.config.pb_c_base,
                    self.config.pb_c_init,
                )
                actions = torch.from_numpy(actions_array).to(
                    device=device,
                ).reshape(root_count, 1)
                resets = torch.from_numpy(resets_array).to(device=device)
                state_slots = torch.from_numpy(slots_array).to(device=device)
                states = state_store[root_indices, state_slots]
                hidden = (
                    hidden_store[root_indices, state_slots].unsqueeze(0),
                    cell_store[root_indices, state_slots].unsqueeze(0),
                )
                (
                    next_states,
                    next_hidden,
                    value_prefixes,
                    values,
                    policy_logits,
                ) = evaluator.evaluate_tensors(
                    states,
                    actions,
                    hidden,
                    resets,
                )
                evaluator.validate_policy(policy_logits, root_count)
                next_slot = simulation + 1
                state_store[:, next_slot].copy_(next_states)
                hidden_store[:, next_slot].copy_(next_hidden[0][0])
                cell_store[:, next_slot].copy_(next_hidden[1][0])

                output_rows = torch.cat(
                    (
                        value_prefixes[:, None],
                        values[:, None],
                        policy_logits,
                    ),
                    dim=1,
                ).float().cpu().numpy()
                tree.expand_and_back_up_arrays(
                    next_slot,
                    output_rows[:, 0],
                    output_rows[:, 1],
                    output_rows[:, 2:],
                )

        return SearchBatchResult(
            visit_counts=tree.visit_counts_array(),
            root_values=tree.root_values_array(),
        )

    def materialize_results(
        self,
        batch: SearchBatchResult,
        *,
        temperature: float = 1.0,
    ) -> tuple[SearchResult, ...]:
        """Sample actions and create per-root objects at the replay boundary."""
        self._validate_temperature(temperature)
        visit_counts = batch.visit_counts.tolist()
        root_values = batch.root_values.tolist()
        return tuple(
            SearchResult(
                action=self.rng.choices(
                    range(len(counts)),
                    weights=_visit_policy(counts, temperature),
                    k=1,
                )[0],
                visit_counts=tuple(counts),
                root_value=value,
            )
            for counts, value in zip(
                visit_counts,
                root_values,
                strict=True,
            )
        )

    @staticmethod
    def _validate_temperature(temperature: float) -> None:
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")

    def _add_root_noise(self, priors: np.ndarray) -> None:
        samples = np.fromiter(
            (
                self.rng.gammavariate(
                    self.config.dirichlet_alpha,
                    1.0,
                )
                for _ in range(priors.shape[0])
            ),
            dtype=np.float64,
            count=priors.shape[0],
        )
        total = float(samples.sum())
        if total > 0.0:
            samples /= total
        else:
            samples.fill(1.0 / priors.shape[0])
        fraction = self.config.root_exploration_fraction
        priors *= 1.0 - fraction
        priors += fraction * samples

@contextmanager
def _evaluator_inference(evaluator: PackedEvaluator):
    """Disable autograd and temporarily put module evaluators in eval mode."""
    module = evaluator if isinstance(evaluator, nn.Module) else None
    was_training = module.training if module is not None else False
    if module is not None:
        module.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        if module is not None:
            module.train(was_training)


def _visit_policy(visits: Sequence[int], temperature: float) -> tuple[float, ...]:
    if len(visits) == 0 or sum(visits) <= 0:
        raise ValueError("at least one action must have been visited")
    best_action = max(range(len(visits)), key=visits.__getitem__)
    if temperature == 0.0:
        return tuple(
            1.0 if action == best_action else 0.0
            for action in range(len(visits))
        )

    log_weights = [
        math.log(visit) / temperature if visit > 0 else -math.inf
        for visit in visits
    ]
    maximum = max(log_weights)
    weights = [
        math.exp(log_weight - maximum) if math.isfinite(log_weight) else 0.0
        for log_weight in log_weights
    ]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


__all__ = [
    "MCTS",
    "MCTSConfig",
    "PackedEvaluator",
    "SearchBatchResult",
    "SearchResult",
]
