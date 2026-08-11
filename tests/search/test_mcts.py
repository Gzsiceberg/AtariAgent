import random

import pytest
import torch

from atariagent.search import Evaluation, MCTS, MCTSConfig, MinMaxStats, Node


class SampleLastRandom(random.Random):
    """Choose the last positive-weight action when sampling a final action."""

    def choices(
        self,
        population,
        weights=None,
        *,
        cum_weights=None,
        k=1,
    ):
        assert weights is not None
        positive = [item for item, weight in zip(population, weights) if weight > 0]
        return [positive[-1]] * k


class RandomPolicyDefaultValue:
    """A random policy with the default zero value/value-prefix prediction."""

    def __init__(self, action_space_size: int, seed: int = 0) -> None:
        self.action_space_size = action_space_size
        self.rng = random.Random(seed)
        self.calls = 0

    def __call__(self, state: int, action: int, hidden: object) -> Evaluation:
        self.calls += 1
        return Evaluation(
            state=state + 1,
            value_prefix=0.0,
            value=0.0,
            policy_logits=[
                self.rng.uniform(-1.0, 1.0)
                for _ in range(self.action_space_size)
            ],
        )


def test_default_mcts_with_random_policy_and_zero_value() -> None:
    evaluator = RandomPolicyDefaultValue(action_space_size=4)
    root = Evaluation(
        state=0,
        value_prefix=0.0,
        value=0.0,
        policy_logits=[0.0, 0.0, 0.0, 0.0],
    )

    result = MCTS(rng=random.Random(7)).search(root, evaluator)

    assert evaluator.calls == 50
    assert sum(result.visit_counts) == 50
    assert not hasattr(result, "root")
    assert result.action in range(4)
    assert result.root_value == pytest.approx(0.0)
    assert not hasattr(result, "policy")
    assert MCTSConfig().value_prefix_horizon == 5


def test_node_recovers_rewards_from_value_prefixes() -> None:
    root = Node(prior=1.0)
    root.expand(Evaluation("root", 0.0, 0.0, [0.0]))
    child = root.children[0]
    child.expand(Evaluation("child", 2.0, 0.0, [0.0]))
    grandchild = child.children[0]
    grandchild.value_prefix = 5.0

    assert child.reward() == pytest.approx(2.0)
    assert grandchild.reward() == pytest.approx(3.0)

    child.reset_value_prefix = True
    assert grandchild.reward() == pytest.approx(5.0)


def test_value_prefix_horizon_resets_evaluator_hidden_state() -> None:
    hidden_inputs: list[int | None] = []

    def evaluator(state: int, action: int, hidden: int | None) -> Evaluation:
        hidden_inputs.append(hidden)
        prefix = (hidden or 0) + 1
        return Evaluation(state + 1, prefix, 0.0, [0.0], prefix)

    config = MCTSConfig(
        num_simulations=3,
        discount=1.0,
        value_prefix_horizon=2,
    )
    root = Evaluation(0, 0.0, 0.0, [0.0], 0)

    result = MCTS(config, rng=random.Random(0)).search(root, evaluator)

    assert hidden_inputs == [0, 1, None]
    assert result.root_value == pytest.approx(1.5)


def test_soft_min_max_normalization_uses_minimum_delta() -> None:
    stats = MinMaxStats(value_delta_max=0.01)
    stats.update(1.0)
    stats.update(1.001)

    assert stats.normalize(1.0) == pytest.approx(0.0)
    assert stats.normalize(1.001) == pytest.approx(0.1)

    stats.clear()
    stats.update(0.5)
    assert stats.normalize(0.5) == pytest.approx(0.5)


def test_temperature_zero_returns_greedy_action() -> None:
    evaluator = RandomPolicyDefaultValue(action_space_size=2)
    result = MCTS(
        MCTSConfig(num_simulations=4), rng=random.Random(0)
    ).search(
        Evaluation(0, 0.0, 0.0, [2.0, -2.0]),
        evaluator,
        temperature=0.0,
    )

    assert result.action == max(
        range(len(result.visit_counts)),
        key=result.visit_counts.__getitem__,
    )


def test_temperature_policy_is_used_to_sample_action() -> None:
    evaluator = RandomPolicyDefaultValue(action_space_size=2)
    result = MCTS(
        MCTSConfig(num_simulations=2), rng=SampleLastRandom(0)
    ).search(
        Evaluation(0, 0.0, 0.0, [0.0, 0.0]),
        evaluator,
        temperature=1.0,
    )

    assert result.visit_counts == (1, 1)
    assert result.action == 1


def test_search_batch_evaluates_all_roots_in_one_call_per_simulation() -> None:
    calls: list[tuple[list[int], list[int], list[object]]] = []

    def evaluator(states, actions, hidden_states):
        calls.append((list(states), list(actions), list(hidden_states)))
        return [
            Evaluation(state + 1, 0.0, 0.0, [0.0])
            for state in states
        ]

    config = MCTSConfig(num_simulations=4)
    roots = [Evaluation(state, 0.0, 0.0, [0.0]) for state in range(3)]
    results = MCTS(config, rng=random.Random(0)).search_batch(roots, evaluator)

    assert len(calls) == config.num_simulations
    assert all(len(states) == len(roots) for states, _, _ in calls)
    assert len(results) == len(roots)
    assert all(result.visit_counts == (4,) for result in results)


@pytest.mark.parametrize("seed", [0, 1, 4, 17])
def test_native_batch_search_matches_python_reference(seed: int) -> None:
    config = MCTSConfig(
        num_simulations=20,
        discount=0.9,
        value_prefix_horizon=3,
    )
    roots = [
        Evaluation(index, 0.0, float(index), [1.0, 0.2, -0.7], 0.0)
        for index in range(3)
    ]

    def evaluator(states, actions, hidden_states):
        return [
            Evaluation(
                state * 3 + action + 1,
                float(hidden or 0.0) + (action + 1) * 0.25,
                float(state + action) / 10.0,
                [0.3 - action, 0.1 + action, -0.2],
                float(hidden or 0.0) + (action + 1) * 0.25,
            )
            for state, action, hidden in zip(
                states,
                actions,
                hidden_states,
                strict=True,
            )
        ]

    native = MCTS(config, rng=random.Random(seed)).search_batch(
        roots,
        evaluator,
        add_exploration_noise=True,
        _deterministic_ties=True,
    )
    reference = MCTS(config, rng=random.Random(seed))._search_batch_python(
        roots,
        evaluator,
        add_exploration_noise=True,
        _deterministic_ties=True,
    )

    assert [result.visit_counts for result in native] == [
        result.visit_counts for result in reference
    ]
    assert [result.root_value for result in native] == pytest.approx(
        [result.root_value for result in reference],
        rel=1e-5,
        abs=1e-6,
    )


def test_module_evaluator_uses_inference_and_eval_modes() -> None:
    class ModuleEvaluator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grad_enabled: list[bool] = []
            self.training_modes: list[bool] = []

        def forward(self, state, action, hidden):
            self.grad_enabled.append(torch.is_grad_enabled())
            self.training_modes.append(self.training)
            return Evaluation(state + 1, 0.0, 0.0, [0.0])

    evaluator = ModuleEvaluator()
    assert evaluator.training

    MCTS(MCTSConfig(num_simulations=2)).search(
        Evaluation(0, 0.0, 0.0, [0.0]), evaluator
    )

    assert evaluator.grad_enabled == [False, False]
    assert evaluator.training_modes == [False, False]
    assert evaluator.training


def test_root_prediction_remains_the_first_value_estimate() -> None:
    def evaluator(state, action, hidden):
        return Evaluation(state + 1, 0.0, 0.0, [0.0])

    result = MCTS(MCTSConfig(num_simulations=1, discount=1.0)).search(
        Evaluation(0, 0.0, 10.0, [0.0]), evaluator
    )

    assert sum(result.visit_counts) == 1
    assert result.root_value == pytest.approx(5.0)
