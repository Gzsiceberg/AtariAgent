import pytest
import torch

from atariagent.training import LearnerProfiler


def test_learner_profiler_excludes_warmup_and_reports_mean_and_p95() -> None:
    profiler = LearnerProfiler(
        warmup_steps=1,
        report_every=3,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert profiler.record({"update_total": 1000.0}) is None
    assert profiler.record(
        {"update_total": 10.0, "replay_sample": 2.0}
    ) is None
    assert profiler.record(
        {"update_total": 20.0, "replay_sample": 4.0}
    ) is None
    summary = profiler.record(
        {"update_total": 30.0, "replay_sample": 6.0}
    )

    assert summary is not None
    assert summary.updates == 3
    assert summary.mean_update_ms == pytest.approx(20.0)
    assert summary.p95_update_ms == pytest.approx(30.0)
    assert summary.updates_per_second == pytest.approx(50.0)
    assert summary.samples_per_second == pytest.approx(800.0)
    assert summary.mean_sections_ms["replay_sample"] == pytest.approx(4.0)
    assert summary.peak_allocated_mib == 0.0
