"""Native packed PUCT and Gumbel tree search for EfficientZero."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from ._tree_search_native import BatchTree as NativeBatchTree

PackedHidden = tuple[Tensor, Tensor]
PackedEvaluation = tuple[Tensor, PackedHidden, Tensor, Tensor, Tensor]


@runtime_checkable
class PackedEvaluator(Protocol):
    """Typed network interface bound to one packed tree-search instance."""

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
class SearchConfig:
    """Configuration shared by PUCT and EfficientZeroV2 Gumbel search."""

    num_simulations: int = 50
    discount: float = 0.997
    pb_c_init: float = 1.25
    pb_c_base: float = 19652.0
    value_delta_max: float = 0.01
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    value_prefix_horizon: int = 5
    search_algorithm: str = "puct"
    num_top_actions: int = 4
    c_visit: float = 50.0
    c_scale: float = 0.1

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
        # Preserve checkpoints and callers created before the conventional
        # search mode was given its precise PUCT name.
        if self.search_algorithm == "mcts":
            object.__setattr__(self, "search_algorithm", "puct")
        if self.search_algorithm not in {"puct", "gumbel"}:
            raise ValueError("search_algorithm must be puct or gumbel")
        if self.num_top_actions < 2 or (
            self.num_top_actions & (self.num_top_actions - 1)
        ):
            raise ValueError("num_top_actions must be a power of two >= 2")
        if self.search_algorithm == "gumbel" and (
            self.num_simulations < self.num_top_actions
        ):
            raise ValueError("Gumbel num_simulations must be >= num_top_actions")
        if not math.isfinite(self.c_visit) or self.c_visit < 0.0:
            raise ValueError("c_visit must be finite and non-negative")
        if not math.isfinite(self.c_scale) or self.c_scale <= 0.0:
            raise ValueError("c_scale must be finite and positive")


@dataclass(frozen=True, slots=True, eq=False)
class SearchResult:
    """Materialized root output, including an optional improved policy."""

    action: int
    visit_counts: NDArray[np.int32]
    root_value: float
    policy_target: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        counts = np.asarray(self.visit_counts, dtype=np.int32)
        if counts.ndim != 1 or counts.shape[0] == 0:
            raise ValueError("visit_counts must be a non-empty 1D array")
        if not counts.flags.c_contiguous:
            counts = np.ascontiguousarray(counts)
        object.__setattr__(self, "visit_counts", counts)
        if self.policy_target is not None:
            policy = np.asarray(self.policy_target, dtype=np.float32)
            if policy.shape != counts.shape or not np.isfinite(policy).all():
                raise ValueError("policy_target must match finite visit_counts")
            if np.any(policy < 0.0) or not np.isclose(policy.sum(), 1.0):
                raise ValueError("policy_target must be a probability distribution")
            object.__setattr__(
                self,
                "policy_target",
                np.ascontiguousarray(policy),
            )

    @property
    def target_policy(self) -> NDArray[np.float32]:
        """Return the search policy target used by replay."""
        if self.policy_target is not None:
            return self.policy_target
        total = int(self.visit_counts.sum())
        if total <= 0:
            raise ValueError("search result must contain visited actions")
        return np.ascontiguousarray(
            self.visit_counts.astype(np.float32) / total,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return (
            self.action == other.action
            and self.root_value == other.root_value
            and np.array_equal(self.visit_counts, other.visit_counts)
            and (
                (self.policy_target is None and other.policy_target is None)
                or (
                    self.policy_target is not None
                    and other.policy_target is not None
                    and np.array_equal(self.policy_target, other.policy_target)
                )
            )
        )


@dataclass(frozen=True, slots=True)
class SearchBatchResult:
    """Contiguous native search output before per-root materialization."""

    visit_counts: NDArray[np.int32]
    root_values: NDArray[np.float32]
    policy_targets: NDArray[np.float32] | None = None
    selected_actions: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        if self.visit_counts.ndim != 2:
            raise ValueError("visit_counts must have shape (roots, actions)")
        if self.root_values.shape != (self.visit_counts.shape[0],):
            raise ValueError("root_values must have shape (roots,)")
        if self.policy_targets is not None and (
            self.policy_targets.shape != self.visit_counts.shape
        ):
            raise ValueError("policy_targets must match visit_counts")
        if self.selected_actions is not None and (
            self.selected_actions.shape != (self.visit_counts.shape[0],)
        ):
            raise ValueError("selected_actions must have shape (roots,)")


class TreeSearch:
    """Run the configured native batched search with a network evaluator."""

    def __init__(
        self,
        config: SearchConfig | None = None,
        *,
        evaluator: PackedEvaluator,
        rng: random.Random | None = None,
    ) -> None:
        if not isinstance(evaluator, PackedEvaluator):
            raise TypeError("evaluator must implement PackedEvaluator")
        self.config = config or SearchConfig()
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
                policy_targets=(
                    np.empty(
                        (0, root_policy_logits.shape[-1]),
                        dtype=np.float32,
                    )
                    if self.config.search_algorithm == "gumbel"
                    else None
                ),
                selected_actions=(
                    np.empty(0, dtype=np.int64)
                    if self.config.search_algorithm == "gumbel"
                    else None
                ),
            )
        if root_values.shape != (root_count,):
            raise ValueError("root_values must have shape (batch_size,)")
        if root_policy_logits.ndim != 2 or root_policy_logits.shape[0] != root_count:
            raise ValueError("root_policy_logits must have shape (batch_size, actions)")
        if root_states.device != root_values.device or (
            root_states.device != root_policy_logits.device
        ):
            raise ValueError("root tensors must be on the same device")
        evaluator = self.evaluator

        # Pack the CPU-tree inputs so CUDA performs one device-to-host
        # transfer and synchronization instead of one for each source tensor.
        root_rows = (
            torch.cat(
                (root_values[:, None], root_policy_logits),
                dim=1,
            )
            .float()
            .cpu()
            .numpy()
        )
        if not np.isfinite(root_rows).all():
            raise ValueError("root values and policy logits must be finite")
        root_logits = root_rows[:, 1:].astype(np.float64)
        root_logits -= root_logits.max(axis=1, keepdims=True)
        root_priors = np.exp(root_logits)
        root_priors /= root_priors.sum(axis=1, keepdims=True)
        root_priors = np.ascontiguousarray(root_priors, dtype=np.float32)
        if add_exploration_noise and self.config.search_algorithm == "puct":
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
            self.config.search_algorithm,
            self.config.num_top_actions,
            self.config.c_visit,
            self.config.c_scale,
            add_exploration_noise,
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
                actions = (
                    torch.from_numpy(actions_array)
                    .to(
                        device=device,
                    )
                    .reshape(root_count, 1)
                )
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

                output_rows = (
                    torch.cat(
                        (
                            value_prefixes[:, None],
                            values[:, None],
                            policy_logits,
                        ),
                        dim=1,
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                tree.expand_and_back_up_arrays(
                    next_slot,
                    output_rows[:, 0],
                    output_rows[:, 1],
                    output_rows[:, 2:],
                )

        is_gumbel = self.config.search_algorithm == "gumbel"
        return SearchBatchResult(
            visit_counts=tree.visit_counts_array(),
            root_values=tree.root_values_array(),
            policy_targets=tree.policy_array() if is_gumbel else None,
            selected_actions=(tree.selected_actions_array() if is_gumbel else None),
        )

    def materialize_results(
        self,
        batch: SearchBatchResult,
        *,
        temperature: float = 1.0,
    ) -> tuple[SearchResult, ...]:
        """Sample actions and create per-root objects at the replay boundary."""
        if batch.selected_actions is not None:
            assert batch.policy_targets is not None
            return tuple(
                SearchResult(
                    action=int(action),
                    visit_counts=counts,
                    root_value=float(value),
                    policy_target=policy,
                )
                for counts, value, policy, action in zip(
                    batch.visit_counts,
                    batch.root_values,
                    batch.policy_targets,
                    batch.selected_actions,
                    strict=True,
                )
            )

        self._validate_temperature(temperature)
        policies = _visit_policy(batch.visit_counts, temperature)
        return tuple(
            SearchResult(
                action=self.rng.choices(
                    range(counts.shape[0]),
                    weights=policy,
                    k=1,
                )[0],
                visit_counts=counts,
                root_value=float(value),
            )
            for counts, policy, value in zip(
                batch.visit_counts,
                policies,
                batch.root_values,
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


def _visit_policy(
    visits: NDArray[np.int32],
    temperature: float,
) -> NDArray[np.float64]:
    """Return batched temperature-adjusted visit weights with NumPy."""
    if visits.ndim not in (1, 2) or visits.shape[-1] == 0:
        raise ValueError("visits must have shape (actions,) or (roots, actions)")
    squeeze = visits.ndim == 1
    rows = visits[None, :] if squeeze else visits
    totals = rows.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("at least one action must have been visited")
    if temperature == 0.0:
        policies = np.zeros(rows.shape, dtype=np.float64)
        policies[np.arange(rows.shape[0]), rows.argmax(axis=1)] = 1.0
        return policies[0] if squeeze else policies

    positive = rows > 0
    log_weights = np.full(rows.shape, -np.inf, dtype=np.float64)
    np.log(rows, out=log_weights, where=positive)
    log_weights /= temperature
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.zeros(rows.shape, dtype=np.float64)
    np.exp(log_weights, out=weights, where=positive)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights[0] if squeeze else weights


# Compatibility aliases for existing callers and old checkpoints.
MCTS = TreeSearch
MCTSConfig = SearchConfig


__all__ = [
    "MCTS",
    "MCTSConfig",
    "PackedEvaluator",
    "SearchBatchResult",
    "SearchConfig",
    "SearchResult",
    "TreeSearch",
]
