"""Split target checkpoint persistence and migration from shared-target snapshots."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from atariagent.training.config import target_network_update_due


@pytest.fixture(scope="module")
def train_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_agent.py"
    spec = importlib.util.spec_from_file_location("train_snapshot_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StateHolder:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


class FakeTrainer:
    def __init__(self):
        self.step_count = 1000
        self.consistency_network = torch.nn.Linear(1, 1)
        self.optimizer = StateHolder({"momentum": torch.tensor([0.9])})

    def training_state_dict(self):
        return {"step_count": self.step_count, "optimizer": self.optimizer.state_dict()}

    def load_training_state_dict(self, state):
        self.step_count = state["step_count"]
        self.optimizer.load_state_dict(state["optimizer"])


@pytest.fixture
def saved_snapshot(train_script, tmp_path):
    agent = SimpleNamespace(
        representation_network=torch.nn.Linear(1, 1),
        prediction_network=torch.nn.Linear(1, 1),
        dynamics_network=torch.nn.Linear(1, 1),
        search=SimpleNamespace(rng=random.Random(2)),
    )
    trainer = FakeTrainer()
    replay = StateHolder({"priorities": torch.tensor([1.0, 2.0])})
    policy_state = {"representation.weight": torch.tensor([[2.0]])}
    bootstrap_state = {"representation.weight": torch.tensor([[7.0]])}
    path = tmp_path / "agent_pre_final.pt"
    kwargs = {
        "agent": agent,
        "trainer": trainer,
        "target_state": policy_state,
        "target_version": 900,
        "bootstrap_state": bootstrap_state,
        "bootstrap_version": 800,
        "update": 1000,
        "config": OmegaConf.create({"seed": 2}),
    }
    train_script.save_pre_final_snapshot(
        path,
        **kwargs,
        replay=replay,
        rng_state={"sentinel": 42},
    )
    return path, kwargs, replay


def load_snapshot(train_script, saved_snapshot):
    path, kwargs, replay = saved_snapshot
    return train_script.load_pre_final_snapshot(
        path,
        agent=kwargs["agent"],
        trainer=kwargs["trainer"],
        replay=replay,
        expected_update=1000,
    )


def test_split_snapshot_round_trip_and_resume_schedules(train_script, saved_snapshot):
    path, kwargs, replay = saved_snapshot
    snapshot = torch.load(path, weights_only=False)
    assert snapshot["snapshot_version"] == 2
    assert not path.with_name(f".{path.name}.tmp").exists()
    kwargs["trainer"].step_count = 0
    replay.state = {}
    update, policy, policy_version, bootstrap, bootstrap_version, rng = load_snapshot(
        train_script, saved_snapshot
    )
    assert update == kwargs["trainer"].step_count == 1000
    assert (policy_version, bootstrap_version) == (900, 800)
    torch.testing.assert_close(policy, kwargs["target_state"])
    torch.testing.assert_close(bootstrap, kwargs["bootstrap_state"])
    assert rng == {"sentinel": 42}
    torch.testing.assert_close(replay.state["priorities"], torch.tensor([1.0, 2.0]))
    assert not target_network_update_due(1001, last_update=policy_version, interval=200)
    assert target_network_update_due(1001, last_update=bootstrap_version, interval=200)


def test_legacy_snapshot_initializes_both_paths_from_shared_target(
    train_script, saved_snapshot
):
    path, kwargs, _ = saved_snapshot
    snapshot = torch.load(path, weights_only=False)
    snapshot["snapshot_version"] = 1
    del snapshot["bootstrap_target_network"]
    del snapshot["bootstrap_target_version"]
    torch.save(snapshot, path)
    _, policy, policy_version, bootstrap, bootstrap_version, _ = load_snapshot(
        train_script, saved_snapshot
    )
    assert policy_version == bootstrap_version == 900
    torch.testing.assert_close(policy, kwargs["target_state"])
    torch.testing.assert_close(bootstrap, policy)


@pytest.mark.parametrize(
    "name,value",
    [
        ("bootstrap_target_network", None),
        ("bootstrap_target_version", None),
        ("bootstrap_target_version", True),
        ("bootstrap_target_version", -1),
        ("bootstrap_target_version", 1001),
        ("target_version", -1),
        ("target_version", 1001),
    ],
)
def test_split_snapshot_rejects_invalid_target_metadata(
    train_script,
    saved_snapshot,
    name,
    value,
):
    path, _, _ = saved_snapshot
    snapshot = torch.load(path, weights_only=False)
    snapshot[name] = value
    torch.save(snapshot, path)
    with pytest.raises((TypeError, ValueError), match="target"):
        load_snapshot(train_script, saved_snapshot)


def test_regular_checkpoint_records_both_targets(
    train_script, saved_snapshot, tmp_path
):
    _, kwargs, _ = saved_snapshot
    path = tmp_path / "agent_latest.pt"
    train_script.save_checkpoint(path, **kwargs)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["target_version"] == 900
    assert checkpoint["bootstrap_target_version"] == 800
    torch.testing.assert_close(checkpoint["target_network"], kwargs["target_state"])
    torch.testing.assert_close(
        checkpoint["bootstrap_target_network"], kwargs["bootstrap_state"]
    )


def test_legacy_checkpoint_writer_arguments_use_shared_target(
    train_script, saved_snapshot, tmp_path
):
    _, kwargs, _ = saved_snapshot
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("bootstrap_")}
    path = tmp_path / "agent_latest.pt"
    train_script.save_checkpoint(path, **kwargs)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["target_version"] == checkpoint["bootstrap_target_version"] == 900
    torch.testing.assert_close(
        checkpoint["target_network"], checkpoint["bootstrap_target_network"]
    )
