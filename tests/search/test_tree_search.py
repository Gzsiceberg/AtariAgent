from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from atariagent.search import (
    MCTS,
    MCTSConfig,
    PackedEvaluator,
    SearchConfig,
    TreeSearch,
)


class PackedScalarEvaluator:
    """Tensor evaluator paired with the independent Python reference below."""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def initial_hidden(batch_size, *, device, dtype):
        hidden = torch.zeros(1, batch_size, 1, device=device, dtype=dtype)
        return hidden, torch.zeros_like(hidden)

    def evaluate_tensors(self, states, actions, hidden, resets=None):
        self.calls += 1
        hidden_value = hidden[0]
        if resets is not None:
            hidden_value = hidden_value.masked_fill(
                resets.reshape(1, states.shape[0], 1),
                0.0,
            )
        prefix = hidden_value + (actions.T.unsqueeze(-1) + 1) * 0.25
        flat_actions = actions[:, 0].to(states.dtype)
        flat_states = states[:, 0]
        next_states = (flat_states * 3 + flat_actions + 1).unsqueeze(1)
        values = (flat_states + flat_actions) / 10.0
        logits = torch.stack(
            (0.3 - flat_actions, 0.1 + flat_actions, flat_actions * 0 - 0.2),
            dim=1,
        )
        return next_states, (prefix, prefix), prefix[0, :, 0], values, logits

    @staticmethod
    def validate_policy(policy_logits, batch_size):
        if policy_logits.shape != (batch_size, 3):
            raise ValueError("invalid policy shape")


@dataclass(frozen=True)
class _Evaluation:
    state: int
    value_prefix: float
    value: float
    policy_logits: list[float]
    hidden: float = 0.0


@dataclass
class _Stats:
    minimum_delta: float
    minimum: float = math.inf
    maximum: float = -math.inf

    def normalize(self, value: float) -> float:
        delta = self.maximum - self.minimum
        if delta > 0.0:
            value = (value - self.minimum) / max(delta, self.minimum_delta)
        return min(max(value, 0.0), 1.0)


@dataclass
class _Node:
    prior: float
    parent: _Node | None = None
    action: int = -1
    state: int = 0
    value_prefix: float = 0.0
    hidden: float = 0.0
    reset: bool = False
    visits: int = 0
    value_sum: float = 0.0
    children: list[_Node] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def reward(self) -> float:
        if self.parent is None:
            return 0.0
        if self.parent.reset:
            return self.value_prefix
        return self.value_prefix - self.parent.value_prefix

    def q(self, discount: float) -> float:
        return self.reward() + discount * self.value


def _softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    values = [math.exp(value - maximum) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def _expand(node: _Node, evaluation: _Evaluation) -> None:
    node.state = evaluation.state
    node.value_prefix = evaluation.value_prefix
    node.hidden = evaluation.hidden
    node.children = [
        _Node(prior=prior, parent=node, action=action)
        for action, prior in enumerate(_softmax(evaluation.policy_logits))
    ]


def _reference_batch(
    config: SearchConfig,
    roots: list[_Evaluation],
    *,
    seed: int,
    add_exploration_noise: bool,
) -> tuple[list[tuple[int, ...]], list[float]]:
    """Independent Python PUCT retained only for native differential tests."""
    rng = random.Random(seed)
    nodes: list[_Node] = []
    stats: list[_Stats] = []
    for evaluation in roots:
        root = _Node(prior=1.0)
        _expand(root, evaluation)
        root.visits = 1
        root.value_sum = evaluation.value
        if add_exploration_noise:
            samples = [
                rng.gammavariate(config.dirichlet_alpha, 1.0) for _ in root.children
            ]
            total = sum(samples)
            for child, sample in zip(root.children, samples, strict=True):
                child.prior = (
                    1.0 - config.root_exploration_fraction
                ) * child.prior + config.root_exploration_fraction * sample / total
        nodes.append(root)
        stats.append(_Stats(config.value_delta_max))

    def evaluator(state: int, action: int, hidden: float | None) -> _Evaluation:
        prefix = float(hidden or 0.0) + (action + 1) * 0.25
        return _Evaluation(
            state * 3 + action + 1,
            prefix,
            float(state + action) / 10.0,
            [0.3 - action, 0.1 + action, -0.2],
            prefix,
        )

    for _ in range(config.num_simulations):
        for root, root_stats in zip(nodes, stats, strict=True):
            node = root
            path = [root]
            parent_mean_q = 0.0
            while node.children:
                visited_q = [
                    child.q(config.discount)
                    for child in node.children
                    if child.visits > 0
                ]
                mean_q = (
                    sum(visited_q) / len(visited_q)
                    if node is root and visited_q
                    else 0.0
                    if node is root
                    else (parent_mean_q + sum(visited_q)) / (1 + len(visited_q))
                )
                child_visits = sum(child.visits for child in node.children)
                exploration_scale = config.pb_c_init + math.log(
                    (child_visits + config.pb_c_base + 1.0) / config.pb_c_base
                )
                sqrt_visits = math.sqrt(child_visits)
                scores = []
                for child in node.children:
                    prior_score = (
                        child.prior
                        * sqrt_visits
                        / (1 + child.visits)
                        * exploration_scale
                    )
                    q = child.q(config.discount) if child.visits else mean_q
                    scores.append(root_stats.normalize(q) + prior_score)
                best = max(scores)
                node = next(
                    child
                    for child, score in zip(node.children, scores, strict=True)
                    if abs(score - best) <= 1e-12
                )
                path.append(node)
                parent_mean_q = mean_q

            parent = node.parent
            assert parent is not None
            evaluation = evaluator(
                parent.state,
                node.action,
                None if parent.reset else parent.hidden,
            )
            _expand(node, evaluation)
            node.reset = node.depth % config.value_prefix_horizon == 0
            bootstrap = evaluation.value
            for visited in reversed(path):
                visited.value_sum += bootstrap
                visited.visits += 1
                bootstrap = visited.reward() + config.discount * bootstrap

            root_stats.minimum = math.inf
            root_stats.maximum = -math.inf
            stack = [root]
            while stack:
                current = stack.pop()
                for child in current.children:
                    if child.visits:
                        value = child.q(config.discount)
                        root_stats.minimum = min(root_stats.minimum, value)
                        root_stats.maximum = max(root_stats.maximum, value)
                        stack.append(child)

    return (
        [tuple(child.visits for child in root.children) for root in nodes],
        [root.value for root in nodes],
    )


def _make_search(
    config: SearchConfig | None = None,
    *,
    seed: int = 0,
    evaluator: PackedEvaluator | None = None,
) -> TreeSearch:
    return TreeSearch(
        config,
        evaluator=evaluator or PackedScalarEvaluator(),
        rng=random.Random(seed),
    )


def test_default_config_matches_efficientzero_search() -> None:
    config = SearchConfig()
    assert config.num_simulations == 50
    assert config.value_prefix_horizon == 5
    assert config.search_algorithm == "puct"


def test_legacy_mcts_names_and_mode_remain_compatible() -> None:
    assert MCTS is TreeSearch
    assert MCTSConfig is SearchConfig
    assert SearchConfig(search_algorithm="mcts").search_algorithm == "puct"


def test_tree_search_requires_a_packed_evaluator() -> None:
    with pytest.raises(TypeError, match="PackedEvaluator"):
        TreeSearch(evaluator=object())  # type: ignore[arg-type]


def test_search_batch_evaluates_all_roots_once_per_simulation() -> None:
    evaluator = PackedScalarEvaluator()
    config = SearchConfig(num_simulations=4)
    states = torch.arange(3, dtype=torch.float32).unsqueeze(1)

    results = _make_search(config, evaluator=evaluator).search_batch(
        states,
        torch.zeros(3),
        torch.zeros(3, 3),
    )

    assert evaluator.calls == config.num_simulations
    assert results.visit_counts.shape == (states.shape[0], 3)
    assert (results.visit_counts.sum(axis=1) == config.num_simulations).all()


@pytest.mark.parametrize("seed", [0, 1, 4, 17])
def test_native_batch_search_matches_python_reference(seed: int) -> None:
    config = SearchConfig(
        num_simulations=20,
        discount=0.9,
        value_prefix_horizon=3,
    )
    states = torch.arange(3, dtype=torch.float32).unsqueeze(1)
    root_values = torch.arange(3, dtype=torch.float32)
    root_logits = torch.tensor([[1.0, 0.2, -0.7]]).expand(3, -1)

    native = _make_search(config, seed=seed).search_batch(
        states,
        root_values,
        root_logits,
        add_exploration_noise=True,
        _deterministic_ties=True,
    )
    counts, values = _reference_batch(
        config,
        [_Evaluation(index, 0.0, float(index), [1.0, 0.2, -0.7]) for index in range(3)],
        seed=seed,
        add_exploration_noise=True,
    )

    assert native.visit_counts.tolist() == [list(row) for row in counts]
    assert native.root_values.tolist() == pytest.approx(
        values,
        rel=1e-5,
        abs=1e-6,
    )


def test_gumbel_materialization_uses_direct_action_and_improved_policy() -> None:
    config = SearchConfig(
        num_simulations=8,
        search_algorithm="gumbel",
        num_top_actions=2,
    )
    mcts = _make_search(config)
    batch = mcts.search_batch(
        torch.zeros(1, 1),
        torch.zeros(1),
        torch.tensor([[1.0, 0.2, -0.7]]),
    )
    result = mcts.materialize_results(batch, temperature=10.0)[0]

    assert result.action == int(batch.selected_actions[0])
    np.testing.assert_array_equal(result.policy_target, batch.policy_targets[0])
    np.testing.assert_array_equal(result.target_policy, batch.policy_targets[0])


def test_temperature_zero_returns_greedy_action() -> None:
    mcts = _make_search(SearchConfig(num_simulations=7))
    batch = mcts.search_batch(
        torch.zeros(2, 1),
        torch.zeros(2),
        torch.tensor([[2.0, -2.0, -3.0]]).expand(2, -1),
    )
    results = mcts.materialize_results(batch, temperature=0.0)
    assert all(isinstance(result.visit_counts, np.ndarray) for result in results)
    assert all(result.visit_counts.dtype == np.int32 for result in results)
    assert all(
        result.action
        == max(range(len(result.visit_counts)), key=result.visit_counts.__getitem__)
        for result in results
    )


def test_module_evaluator_uses_inference_and_eval_modes() -> None:
    class ModuleEvaluator(torch.nn.Module, PackedScalarEvaluator):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            PackedScalarEvaluator.__init__(self)
            self.grad_enabled: list[bool] = []
            self.training_modes: list[bool] = []

        def evaluate_tensors(self, states, actions, hidden, resets=None):
            self.grad_enabled.append(torch.is_grad_enabled())
            self.training_modes.append(self.training)
            return super().evaluate_tensors(states, actions, hidden, resets)

    evaluator = ModuleEvaluator()
    _make_search(
        SearchConfig(num_simulations=2),
        evaluator=evaluator,
    ).search_batch(
        torch.zeros(2, 1),
        torch.zeros(2),
        torch.zeros(2, 3),
    )

    assert evaluator.grad_enabled == [False, False]
    assert evaluator.training_modes == [False, False]
    assert evaluator.training
