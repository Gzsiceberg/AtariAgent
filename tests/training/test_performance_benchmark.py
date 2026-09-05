"""CPU-only checks for benchmark instrumentation (no production behavior changes)."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def benchmark():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "performance_benchmark", scripts / "benchmark_training_performance.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


def test_distribution(benchmark):
    assert benchmark.distribution([]) == {}
    result = benchmark.distribution([1, 2, 3])
    assert result == {"count": 3, "mean": 2, "p50": 2, "p95": 2.9, "max": 3}


def test_timed_lock_exclusion_and_exception(benchmark):
    measurements = benchmark.Measurements()
    lock = benchmark.TimedLock(measurements)
    inside = Lock()

    def acquire(_):
        with lock:
            assert inside.acquire(blocking=False)
            inside.release()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire, range(100)))
    with pytest.raises(RuntimeError), lock:
        raise RuntimeError("test")
    with lock:
        pass
    assert len(measurements.values["complete_lock_wait_ms"]) == 102
    assert len(measurements.values["complete_lock_hold_ms"]) == 102
    assert not lock.lock.locked()


def test_priority_measurement_preserves_values_and_errors(benchmark):
    replay = benchmark.MeasuredReplay(10, unroll_steps=1, td_steps=1)
    replay.measurements = benchmark.Measurements()
    replay.add(
        benchmark.make_trajectory(
            sampleable_steps=5,
            lookahead_steps=1,
            block_id=0,
            stack_size=4,
            channels=1,
            screen_size=8,
            action_space_size=3,
        )
    )
    indices = torch.tensor(replay._transition_ids[:2].copy())
    priorities = torch.tensor([2.0, 3.0])
    replay.update_priorities(indices, priorities)
    np.testing.assert_array_equal(replay._priorities[:2], [2.0, 3.0])
    with pytest.raises(KeyError):
        replay.update_priorities(torch.tensor([-999]), torch.tensor([1.0]))
    assert len(replay.measurements.values["priority_d2h_and_stream_wait_ms"]) == 2
