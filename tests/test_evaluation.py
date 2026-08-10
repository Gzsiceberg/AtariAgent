import random
from types import SimpleNamespace

import pytest

from atariagent.evaluation import (
    EvaluationRecord,
    EvaluationStats,
    evaluate_agent,
    plot_evaluation_history,
    write_evaluation_history,
)


class OneStepEnvironment:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(n=2)
        self.reward = 0.0
        self.closed = False

    def reset(self, *, seed=None):
        self.reward = float(seed)
        return [0], {}

    def step(self, action):
        assert action == 1
        return [0], self.reward, True, False, {}

    def close(self):
        self.closed = True


class GreedyAgent:
    def __init__(self) -> None:
        self.mcts = SimpleNamespace(rng=random.Random(123))

    def act(self, observations, **kwargs):
        assert kwargs == {
            "add_exploration_noise": False,
            "temperature": 0.0,
        }
        return SimpleNamespace(actions=(1,))


def test_evaluate_agent_reports_episode_reward_statistics() -> None:
    environment = OneStepEnvironment()
    agent = GreedyAgent()
    rng_state = agent.mcts.rng.getstate()

    stats = evaluate_agent(agent, lambda: environment, episodes=3, seed=1)

    assert stats.rewards == (1.0, 2.0, 3.0)
    assert stats.mean == pytest.approx(2.0)
    assert stats.median == pytest.approx(2.0)
    assert stats.std == pytest.approx(0.81649658)
    assert environment.closed
    assert agent.mcts.rng.getstate() == rng_state


def test_evaluation_history_writes_json_and_plot(tmp_path) -> None:
    stats = EvaluationStats.from_rewards((1.0, 3.0))
    records = [EvaluationRecord.create(100, tmp_path / "model.pt", stats)]
    data_path = tmp_path / "nested" / "evaluation.json"
    plot_path = tmp_path / "nested" / "evaluation.png"

    write_evaluation_history(data_path, records, environment_id="ALE/Pong-v5")
    plot_evaluation_history(plot_path, records, title="Pong")

    assert '"update": 100' in data_path.read_text()
    assert '"rewards": [' in data_path.read_text()
    assert plot_path.stat().st_size > 0
