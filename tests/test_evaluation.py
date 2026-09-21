import random
from types import SimpleNamespace

import pytest
import torch

import atariagent.evaluation as evaluation_module
from atariagent.evaluation import (
    EvaluationRecord,
    EvaluationStats,
    evaluate_agent,
    load_agent_checkpoint,
    plot_evaluation_history,
    write_evaluation_history,
)
from atariagent.typecheck import runtime_typechecking_enabled


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
            "temperature": 0.0,
            "root_noise_temperature": 0.0,
            "gumbel_sampling": False,
        }
        return SimpleNamespace(actions=(1,))


class VariableLengthEnvironment:
    def __init__(self) -> None:
        self.remaining_steps = 0
        self.closed = False

    def reset(self, *, seed=None):
        self.remaining_steps = int(seed) % 3 + 1
        return [seed], {}

    def step(self, action):
        assert action == 1
        self.remaining_steps -= 1
        return [0], 1.0, self.remaining_steps == 0, False, {}

    def close(self):
        self.closed = True


class BatchedGreedyAgent:
    def __init__(self) -> None:
        self.mcts = SimpleNamespace(rng=random.Random(123))
        self.batch_sizes: list[int] = []

    def act(self, observations, **kwargs):
        self.batch_sizes.append(len(observations))
        return SimpleNamespace(actions=(1,) * len(observations))


def test_evaluate_agent_reports_episode_reward_statistics() -> None:
    environment = OneStepEnvironment()
    agent = GreedyAgent()
    rng_state = agent.mcts.rng.getstate()

    stats = evaluate_agent(
        agent, lambda: environment, episodes=3, num_envs=1, seed=1
    )

    assert stats.rewards == (1.0, 2.0, 3.0)
    assert stats.mean == pytest.approx(2.0)
    assert stats.median == pytest.approx(2.0)
    assert stats.std == pytest.approx(0.81649658)
    assert environment.closed
    assert agent.mcts.rng.getstate() == rng_state
    assert runtime_typechecking_enabled()


def test_evaluate_agent_batches_parallel_environments() -> None:
    environments: list[VariableLengthEnvironment] = []

    def factory() -> VariableLengthEnvironment:
        environment = VariableLengthEnvironment()
        environments.append(environment)
        return environment

    agent = BatchedGreedyAgent()
    stats = evaluate_agent(agent, factory, episodes=5, num_envs=2, seed=0)

    assert stats.rewards == (1.0, 2.0, 3.0, 1.0, 2.0)
    assert len(environments) == 2
    assert all(environment.closed for environment in environments)
    assert max(agent.batch_sizes) == 2
    assert agent.batch_sizes[-1] == 1


def test_evaluate_agent_rejects_invalid_parallelism() -> None:
    with pytest.raises(ValueError, match="num_envs"):
        evaluate_agent(GreedyAgent(), OneStepEnvironment, episodes=1, num_envs=0)


@pytest.mark.parametrize("precision", [None, "fp32", "bf16"])
@pytest.mark.parametrize("action_embedding", [None, True, False])
def test_load_agent_checkpoint_applies_search_overrides(monkeypatch, precision, action_embedding) -> None:
    checkpoint = {
        "config": {
            "environment": {
                "frame_stack": 4,
                "frame_skip": 4,
                "grayscale": True,
            },
            "self_play": {
                "num_simulations": 16,
                "search_algorithm": "gumbel",
            },
            "training": {"discount": 0.997, "lstm_horizon": 5},
        },
        "representation": {},
        "dynamics": {},
        "prediction": {},
    }

    if precision is not None:
        checkpoint["config"]["training"]["precision"] = precision
    if action_embedding is not None:
        checkpoint["config"]["model"] = {"action_embedding": action_embedding}

    class FakeNetwork:
        def load_state_dict(self, state_dict) -> None:
            assert state_dict == {}

    class FakeAgent:
        def __init__(self, in_channels, action_space_size, *, search_config, precision, action_embedding) -> None:
            assert in_channels == 4
            self.action_embedding = action_embedding
            self.precision = precision
            self.action_space_size = action_space_size
            self.search_config = search_config
            self.representation_network = FakeNetwork()
            self.dynamics_network = FakeNetwork()
            self.prediction_network = FakeNetwork()

        def to(self, device):
            assert device == torch.device("cpu")
            return self

        def eval(self) -> None:
            pass

    monkeypatch.setattr(
        evaluation_module.torch, "load", lambda *args, **kwargs: checkpoint
    )
    monkeypatch.setattr(evaluation_module, "AtariAgent", FakeAgent)

    agent, saved_config = load_agent_checkpoint(
        "checkpoint.pt",
        action_space_size=6,
        device=torch.device("cpu"),
        search_algorithm_override="puct",
        num_simulations_override=50,
    )

    assert agent.action_space_size == 6
    assert agent.action_embedding is (True if action_embedding is None else action_embedding)
    assert agent.precision == (precision or "fp32")
    assert agent.search_config.search_algorithm == "puct"
    assert agent.search_config.num_simulations == 50
    assert saved_config is checkpoint["config"]

    gumbel_agent, _ = load_agent_checkpoint(
        "checkpoint.pt",
        action_space_size=18,
        device=torch.device("cpu"),
    )

    assert gumbel_agent.search_config.search_algorithm == "gumbel"
    assert gumbel_agent.search_config.num_simulations == 16
    assert gumbel_agent.search_config.num_top_actions == 8


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
