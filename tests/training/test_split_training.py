"""Exercise collection, independent publication, and final-phase resume on CPU."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir


class TinyEnvironment:
    action_space = SimpleNamespace(n=2)

    def reset(self, *, seed=None):
        self.step_count = 0
        return self.observation(), {}

    def observation(self):
        return np.full((4, 96, 96, 1), self.step_count, dtype=np.uint8)

    def step(self, action):
        self.step_count += 1
        return self.observation(), 1.0, self.step_count == 5, False, {}

    def close(self):
        pass


def test_training_publishes_split_targets_and_resumes_them(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "train_split_test", root / "scripts/train_agent.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    # Production intentionally requires CUDA; this small integration test uses
    # the same learner/reanalysis code with a CPU device and a synthetic env.
    monkeypatch.setattr(script, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(script, "configure_training_backend", lambda *a, **kw: None)
    monkeypatch.setattr(script, "create_environments", lambda _: [TinyEnvironment()])
    monkeypatch.setattr(
        script,
        "capture_rng_state",
        lambda: {
            "python": script.random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": [],
        },
    )
    publications = []
    original_pipeline = script.ReanalysisPipeline

    class RecordingPipeline(original_pipeline):
        def publish_weights(
            self, version, state, *, wait=False, policy=True, bootstrap=True
        ):
            super().publish_weights(
                version, state, wait=wait, policy=policy, bootstrap=bootstrap
            )
            publications.append((version, policy, bootstrap))

    monkeypatch.setattr(script, "ReanalysisPipeline", RecordingPipeline)
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        config = compose(
            config_name="train_agent",
            overrides=[
                "environment.grayscale=true",
                "self_play.num_envs=1",
                "self_play.total_transitions=10",
                "self_play.steps_per_iteration=2",
                "self_play.trajectory_length=5",
                "self_play.num_simulations=1",
                "replay.max_transitions=10",
                "replay.warmup_transitions=2",
                "training.steps=10",
                "training.final_steps=3",
                "training.batch_size=2",
                "training.unroll_steps=1",
                "training.td_steps=1",
                "training.lstm_horizon=1",
                "training.precision=fp32",
                "training.compile_model=false",
                "training.progress_mode=never",
                "training.log_every=1",
                "training.batch_max_in_flight=1",
                "training.batch_ready_prefetch=1",
                "reanalysis.prefetch_batches=1",
                "reanalysis.worker_num_threads=1",
                "reanalysis.policy_chunk_size=4",
                "reanalysis.policy_update_interval=4",
                "reanalysis.bootstrap_update_interval=6",
                "reanalysis.cache_target_ttl=0",
                "evaluation.enabled=false",
                "augmentation.enabled=false",
                "checkpoint.collection_interval=10",
                "checkpoint.final_interval=3",
                f"checkpoint.path={tmp_path}/agent_latest.pt",
                f"checkpoint.pre_final_snapshot_path={tmp_path}/agent_pre_final.pt",
            ],
        )
    script.main.__wrapped__(config)
    assert (4, True, False) in publications
    assert (6, False, True) in publications
    assert (8, True, False) in publications
    assert (12, True, True) in publications
    snapshot = torch.load(tmp_path / "agent_pre_final.pt", weights_only=False)
    assert snapshot["target_version"] == 8
    assert snapshot["bootstrap_target_version"] == 6
    final = torch.load(tmp_path / "agent_latest.pt", weights_only=True)
    assert final["update"] == 13
    assert final["target_version"] == final["bootstrap_target_version"] == 12

    publications.clear()
    config.checkpoint.resume_pre_final_path = str(tmp_path / "agent_pre_final.pt")
    config.checkpoint.path = str(tmp_path / "resumed_latest.pt")
    script.main.__wrapped__(config)
    assert publications == [(8, True, False), (6, False, True), (12, True, True)]
    resumed = torch.load(tmp_path / "resumed_latest.pt", weights_only=True)
    assert resumed["update"] == 13
    assert resumed["target_version"] == resumed["bootstrap_target_version"] == 12
    for name in ("representation", "prediction", "dynamics", "consistency"):
        torch.testing.assert_close(resumed[name], final[name])
