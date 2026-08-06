"""Pure-Python Monte Carlo tree search for EfficientZero-style models.

The search consumes an :class:`Evaluation` for the root and a generic
recurrent evaluator for subsequent states.  In particular, recurrent
predictions are *value prefixes*, not one-step rewards.  The reward on an
edge is recovered by subtracting the parent prefix from the child prefix.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import random
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Output required to expand one search node.

    Attributes:
        state: Latent state represented by the node.
        value_prefix: Cumulative predicted reward since the last prefix reset.
        value: Bootstrap value predicted for ``state``.
        policy_logits: One logit for every discrete action.
        value_prefix_hidden: Recurrent state used by value-prefix prediction.
    """

    state: Any
    value_prefix: float
    value: float
    policy_logits: Sequence[float]
    value_prefix_hidden: Any = None


@runtime_checkable
class RecurrentEvaluator(Protocol):
    """Callable used by MCTS to evaluate an action from a latent state.

    ``value_prefix_hidden`` is ``None`` at the start of a value-prefix
    horizon.  This convention matches recurrent modules which initialize a
    zero hidden state when no state is supplied.
    """

    def __call__(
        self, state: Any, action: int, value_prefix_hidden: Any
    ) -> Evaluation: ...


@runtime_checkable
class BatchedRecurrentEvaluator(Protocol):
    """Evaluate one selected leaf for every root in a single call."""

    def __call__(
        self,
        states: Sequence[Any],
        actions: Sequence[int],
        value_prefix_hidden: Sequence[Any],
    ) -> Sequence[Evaluation]: ...


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


@dataclass(slots=True)
class MinMaxStats:
    """Track and softly normalize Q values observed in a search tree."""

    value_delta_max: float = 0.01
    minimum: float = math.inf
    maximum: float = -math.inf

    def __post_init__(self) -> None:
        if self.value_delta_max <= 0.0:
            raise ValueError("value_delta_max must be positive")

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def clear(self) -> None:
        self.minimum = math.inf
        self.maximum = -math.inf

    def normalize(self, value: float) -> float:
        """Normalize with EfficientZero's soft minimum-maximum range."""
        delta = self.maximum - self.minimum
        if delta > 0.0:
            value = (value - self.minimum) / max(delta, self.value_delta_max)
        return min(max(value, 0.0), 1.0)


@dataclass(slots=True)
class Node:
    """A node in the latent-state search tree."""

    prior: float
    parent: Node | None = None
    action: int | None = None
    state: Any = None
    value_prefix: float = 0.0
    value_prefix_hidden: Any = None
    reset_value_prefix: bool = False
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def reward(self) -> float:
        """Recover this edge's reward from consecutive value prefixes."""
        if self.parent is None:
            return 0.0
        if self.parent.reset_value_prefix:
            return self.value_prefix
        return self.value_prefix - self.parent.value_prefix

    def q_value(self, discount: float) -> float:
        return self.reward() + discount * self.value

    def expand(self, evaluation: Evaluation) -> None:
        if self.expanded:
            raise ValueError("cannot expand a node more than once")
        logits = _as_finite_floats(evaluation.policy_logits, "policy_logits")
        if not logits:
            raise ValueError("policy_logits must contain at least one action")

        self.state = evaluation.state
        self.value_prefix = _finite_float(evaluation.value_prefix, "value_prefix")
        self.value_prefix_hidden = evaluation.value_prefix_hidden
        for action, prior in enumerate(_softmax(logits)):
            self.children[action] = Node(prior=prior, parent=self, action=action)

    def add_exploration_noise(
        self,
        alpha: float,
        fraction: float,
        rng: random.Random,
    ) -> None:
        if not self.expanded:
            raise ValueError("node must be expanded before adding noise")
        if fraction == 0.0:
            return
        samples = [rng.gammavariate(alpha, 1.0) for _ in self.children]
        total = sum(samples)
        # gammavariate is positive, but retain a safe fallback for custom RNGs.
        noise = (
            [sample / total for sample in samples]
            if total > 0.0
            else [1.0 / len(samples)] * len(samples)
        )
        for child, sample in zip(self.children.values(), noise, strict=True):
            child.prior = (1.0 - fraction) * child.prior + fraction * sample


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Outputs of a completed root search."""

    action: int
    policy: tuple[float, ...]
    visit_counts: tuple[int, ...]
    root_value: float
    root: Node


class MCTS:
    """Run UCT search over states generated by a recurrent evaluator."""

    def __init__(
        self,
        config: MCTSConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or MCTSConfig()
        self.rng = rng or random.Random()

    def search(
        self,
        root_evaluation: Evaluation,
        evaluator: RecurrentEvaluator | Callable[[Any, int, Any], Evaluation],
        *,
        add_exploration_noise: bool = False,
        temperature: float = 1.0,
    ) -> SearchResult:
        """Search from an already evaluated root latent state.

        The root prediction is counted as its first value estimate. Each of
        ``num_simulations`` then selects one unexpanded node, evaluates it,
        expands it, and backs its value up to the root. The returned action is
        sampled from the temperature-adjusted visit policy; use temperature
        zero for deterministic evaluation.
        """
        self._validate_temperature(temperature)
        root = self._initialize_root(root_evaluation, add_exploration_noise)
        min_max_stats = MinMaxStats(self.config.value_delta_max)

        with _evaluator_inference(evaluator):
            for _ in range(self.config.num_simulations):
                search_path = self._select_path(root, min_max_stats)
                leaf, parent = self._leaf_and_parent(search_path)
                hidden = (
                    None
                    if parent.reset_value_prefix
                    else parent.value_prefix_hidden
                )
                evaluation = evaluator(parent.state, leaf.action, hidden)
                if not isinstance(evaluation, Evaluation):
                    raise TypeError("evaluator must return an Evaluation")
                self._expand_and_back_up(
                    root,
                    min_max_stats,
                    search_path,
                    leaf,
                    evaluation,
                )

        return self._result(root, temperature)

    def search_batch(
        self,
        root_evaluations: Sequence[Evaluation],
        evaluator: BatchedRecurrentEvaluator
        | Callable[
            [Sequence[Any], Sequence[int], Sequence[Any]],
            Sequence[Evaluation],
        ],
        *,
        add_exploration_noise: bool = False,
        temperature: float = 1.0,
    ) -> tuple[SearchResult, ...]:
        """Search roots in parallel with one batched evaluation per simulation."""
        self._validate_temperature(temperature)
        roots = [
            self._initialize_root(root, add_exploration_noise)
            for root in root_evaluations
        ]
        if not roots:
            return ()
        min_max_stats = [
            MinMaxStats(self.config.value_delta_max) for _ in roots
        ]

        with _evaluator_inference(evaluator):
            for _ in range(self.config.num_simulations):
                search_paths = [
                    self._select_path(root, stats)
                    for root, stats in zip(roots, min_max_stats, strict=True)
                ]
                leaves_and_parents = [
                    self._leaf_and_parent(path) for path in search_paths
                ]
                states = [parent.state for _, parent in leaves_and_parents]
                actions = [leaf.action for leaf, _ in leaves_and_parents]
                hidden_states = [
                    None
                    if parent.reset_value_prefix
                    else parent.value_prefix_hidden
                    for _, parent in leaves_and_parents
                ]
                evaluations = evaluator(states, actions, hidden_states)
                if not isinstance(evaluations, Sequence):
                    raise TypeError(
                        "batched evaluator must return a sequence of Evaluations"
                    )
                if len(evaluations) != len(roots):
                    raise ValueError(
                        "batched evaluator must return one Evaluation per root"
                    )

                for root, stats, path, pair, evaluation in zip(
                    roots,
                    min_max_stats,
                    search_paths,
                    leaves_and_parents,
                    evaluations,
                    strict=True,
                ):
                    if not isinstance(evaluation, Evaluation):
                        raise TypeError(
                            "batched evaluator must return Evaluations"
                        )
                    leaf, _ = pair
                    self._expand_and_back_up(
                        root, stats, path, leaf, evaluation
                    )

        return tuple(self._result(root, temperature) for root in roots)

    def _initialize_root(
        self, evaluation: Evaluation, add_exploration_noise: bool
    ) -> Node:
        root_value = _finite_float(evaluation.value, "value")
        root = Node(prior=1.0)
        root.expand(evaluation)
        root.visit_count = 1
        root.value_sum = root_value
        if add_exploration_noise:
            root.add_exploration_noise(
                self.config.dirichlet_alpha,
                self.config.root_exploration_fraction,
                self.rng,
            )
        return root

    @staticmethod
    def _leaf_and_parent(search_path: Sequence[Node]) -> tuple[Node, Node]:
        leaf = search_path[-1]
        parent = leaf.parent
        if parent is None or leaf.action is None:
            raise RuntimeError("selection did not reach a child leaf")
        return leaf, parent

    def _expand_and_back_up(
        self,
        root: Node,
        stats: MinMaxStats,
        search_path: Sequence[Node],
        leaf: Node,
        evaluation: Evaluation,
    ) -> None:
        leaf.expand(evaluation)
        leaf.reset_value_prefix = (
            leaf.depth % self.config.value_prefix_horizon == 0
        )
        self._back_up(search_path, _finite_float(evaluation.value, "value"))
        self._rebuild_min_max(root, stats)

    @staticmethod
    def _validate_temperature(temperature: float) -> None:
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")

    def _result(self, root: Node, temperature: float) -> SearchResult:
        visit_counts = tuple(
            child.visit_count for child in root.children.values()
        )
        policy = _visit_policy(visit_counts, temperature)
        action = self.rng.choices(
            range(len(policy)), weights=policy, k=1
        )[0]
        return SearchResult(
            action=action,
            policy=policy,
            visit_counts=visit_counts,
            root_value=root.value,
            root=root,
        )

    def _select_path(self, root: Node, stats: MinMaxStats) -> list[Node]:
        node = root
        path = [root]
        parent_mean_q = 0.0
        while node.expanded:
            mean_q = self._mean_q(
                node,
                parent_mean_q=parent_mean_q,
                is_root=node is root,
            )
            node = self._select_child(node, mean_q, stats)
            path.append(node)
            parent_mean_q = mean_q
        return path

    def _mean_q(self, node: Node, *, parent_mean_q: float, is_root: bool) -> float:
        visited_q = [
            child.q_value(self.config.discount)
            for child in node.children.values()
            if child.visit_count > 0
        ]
        if is_root:
            return sum(visited_q) / len(visited_q) if visited_q else 0.0
        return (parent_mean_q + sum(visited_q)) / (1 + len(visited_q))

    def _select_child(
        self,
        node: Node,
        mean_q: float,
        stats: MinMaxStats,
    ) -> Node:
        children_visits = sum(child.visit_count for child in node.children.values())
        exploration_scale = (
            self.config.pb_c_init
            + math.log(
                (children_visits + self.config.pb_c_base + 1.0)
                / self.config.pb_c_base
            )
        )
        sqrt_visits = math.sqrt(children_visits)

        best_score = -math.inf
        best_children: list[Node] = []
        for child in node.children.values():
            prior_score = (
                child.prior
                * sqrt_visits
                / (1 + child.visit_count)
                * exploration_scale
            )
            q = (
                child.q_value(self.config.discount)
                if child.visit_count > 0
                else mean_q
            )
            score = stats.normalize(q) + prior_score
            if score > best_score + 1e-12:
                best_score = score
                best_children = [child]
            elif abs(score - best_score) <= 1e-12:
                best_children.append(child)
        return self.rng.choice(best_children)

    def _back_up(self, search_path: Sequence[Node], leaf_value: float) -> None:
        bootstrap_value = leaf_value
        for node in reversed(search_path):
            node.value_sum += bootstrap_value
            node.visit_count += 1
            bootstrap_value = node.reward() + self.config.discount * bootstrap_value

    def _rebuild_min_max(self, root: Node, stats: MinMaxStats) -> None:
        stats.clear()
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                if child.visit_count > 0:
                    stats.update(child.q_value(self.config.discount))
                    stack.append(child)


@contextmanager
def _evaluator_inference(evaluator: Any):
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


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_finite_floats(values: Sequence[float], name: str) -> list[float]:
    try:
        return [_finite_float(value, name) for value in values]
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of scalar numbers") from error


def _softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    exponentials = [math.exp(logit - maximum) for logit in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _visit_policy(visits: Sequence[int], temperature: float) -> tuple[float, ...]:
    if not visits or sum(visits) <= 0:
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
    "BatchedRecurrentEvaluator",
    "Evaluation",
    "MCTS",
    "MCTSConfig",
    "MinMaxStats",
    "Node",
    "RecurrentEvaluator",
    "SearchResult",
]
