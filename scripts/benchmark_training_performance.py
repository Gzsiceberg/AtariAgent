#!/usr/bin/env python3
"""Short production-config systems baseline; synthetic replay, no score claims.

Example: uv run python scripts/benchmark_training_performance.py \
    +benchmark.output=measurements/early +benchmark.replay_size=2000
All benchmark controls live under +benchmark; production config is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock, current_thread
from time import perf_counter

import hydra
import numpy as np
import torch
from atariagent.search._tree_search_native import set_num_threads
from benchmark_replay import make_trajectory
from omegaconf import DictConfig, OmegaConf

# Import registers the production Hydra schema/resolvers and reuses backend setup.
from train_agent import configure_training_backend, create_environments, save_checkpoint

from atariagent.agent import AtariAgent
from atariagent.models import ConsistencyNetwork
from atariagent.replay import FIFOReplayBuffer
from atariagent.search import SearchConfig
from atariagent.training import (
    BatchWorker,
    ReanalysisPipeline,
    Trainer,
    make_target_state,
)
from atariagent.typecheck import set_runtime_typechecking


def distribution(values):
    return (
        {
            "count": len(values),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(max(values)),
        }
        if values
        else {}
    )


class Measurements:
    def __init__(self):
        self.values = defaultdict(list)

    @contextmanager
    def time(self, name):
        torch.cuda.nvtx.range_push(name)
        start = perf_counter()
        try:
            yield
        finally:
            self.values[name].append((perf_counter() - start) * 1000)
            torch.cuda.nvtx.range_pop()


class TimedLock:
    """Benchmark-only drop-in context manager; retain normal lock semantics."""

    def __init__(self, measurements):
        self.lock = Lock()
        self.measurements = measurements

    def __enter__(self):
        start = perf_counter()
        self.lock.acquire()
        self.owner_name = (
            "sample"
            if current_thread().name == "atariagent-batch-worker"
            else "complete"
        )
        self.measurements.values[f"{self.owner_name}_lock_wait_ms"].append(
            (perf_counter() - start) * 1000
        )
        self.start = perf_counter()
        return self

    def __exit__(self, *_):
        self.measurements.values[f"{self.owner_name}_lock_hold_ms"].append(
            (perf_counter() - self.start) * 1000
        )
        self.lock.release()


class MeasuredReplay(FIFOReplayBuffer):
    def sample(self, *args, **kwargs):
        with self.measurements.time("replay_sample_cpu_ms"):
            return super().sample(*args, **kwargs)

    def update_priorities(self, indices, priorities):
        # Deliberately keep D2H INSIDE the caller's lock: unchanged baseline.
        with self.measurements.time("priority_d2h_and_stream_wait_ms"):
            priorities = priorities.detach().cpu().numpy()
        with self.measurements.time("priority_mutation_cpu_ms"):
            return super().update_priorities(indices, priorities)


class MeasuredPipeline(ReanalysisPipeline):
    def wait_next(self):
        ready = super().wait_next()
        for name in (
            "peak_memory_bytes",
            "value_roots_requested",
            "value_roots_searched",
            "value_cache_hits",
        ):
            self.measurements.values[f"native_{name}"].append(getattr(ready, name))
        return ready


@hydra.main(version_base=None, config_path="../configs", config_name="train_agent")
def main(c: DictConfig):
    b = c.get("benchmark", {})
    output = Path(b.get("output", "measurements/training_baseline"))
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    (output / "benchmark_source.py").write_bytes(source)
    replay_size = int(b.get("replay_size", 2000))
    updates = int(b.get("updates", 100))
    warmup = int(b.get("warmup", 10))
    repetitions = int(b.get("repetitions", 3))
    start_step = int(b.get("start_step", 0))
    mode = b.get("mode", "pipeline")
    trace = bool(b.get("trace", False))
    cold_windows = bool(b.get("cold_windows", True))
    pipeline_warmup = int(b.get("pipeline_warmup", warmup))
    if pipeline_warmup < 1:
        raise ValueError("pipeline warmup must be positive")
    if mode not in ("pipeline", "learner") or min(updates, warmup, repetitions) < 1:
        raise ValueError("invalid mode or counts")
    if not c.training.batch_size <= replay_size <= c.replay.max_transitions:
        raise ValueError("replay size must fit production capacity and batch size")
    OmegaConf.save(c, output / "resolved_config.yaml", resolve=True)
    device = torch.device("cuda")
    set_runtime_typechecking(c.training.runtime_type_checks)
    configure_training_backend(
        c.training.deterministic,
        # Retain the original diagnostic override; prefer the production Hydra
        # setting for new comparisons.
        cudnn_benchmark=bool(b.get("cudnn_benchmark", c.training.cudnn_benchmark)),
    )
    set_num_threads(c.reanalysis.worker_num_threads)
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)
    envs = create_environments(c)
    actions = int(envs[0].action_space.n)
    for env in envs:
        env.close()
    channels = 1 if c.environment.grayscale else 3
    discount = c.training.discount**c.environment.frame_skip
    search = SearchConfig(
        num_simulations=c.self_play.num_simulations,
        discount=discount,
        value_prefix_horizon=c.training.lstm_horizon,
        search_algorithm=c.self_play.search_algorithm,
        root_exploration_fraction=c.self_play.root_exploration_fraction,
        c_visit=c.self_play.c_visit,
        c_scale=c.self_play.c_scale,
    )
    agent = AtariAgent(
        c.environment.frame_stack * channels,
        actions,
        search_config=search,
        search_rng=random.Random(c.seed),
    ).to(device)
    training_keys = (
        "optimizer",
        "learning_rate",
        "momentum",
        "weight_decay",
        "lr_warmup_steps",
        "lr_decay_rate",
        "lr_decay_steps",
        "steps",
        "final_steps",
        "unroll_steps",
        "lstm_horizon",
        "max_gradient_norm",
        "precision",
        "compile_model",
        "compile_mode",
    )
    trainer = Trainer(
        agent.representation_network,
        agent.dynamics_network,
        agent.prediction_network,
        consistency_network=ConsistencyNetwork().to(device),
        augmentation=c.augmentation.transforms if c.augmentation.enabled else None,
        augmentation_shift_delta=c.augmentation.shift_delta,
        augmentation_intensity_scale=c.augmentation.intensity_scale,
        image_shape=(c.environment.screen_size, c.environment.screen_size),
        priority_epsilon=c.replay.priority_epsilon,
        **{key: c.training[key] for key in training_keys},
        **dict(c.loss),
    )
    trainer._step_count = start_step
    m = Measurements()
    replay = MeasuredReplay(
        c.replay.max_transitions,
        unroll_steps=c.training.unroll_steps,
        td_steps=c.training.td_steps,
        discount=discount,
        per_mode=c.replay.per_mode,
        priority_alpha=c.replay.priority_alpha,
        priority_beta=c.replay.priority_beta_initial,
        priority_epsilon=c.replay.priority_epsilon,
        seed=c.seed,
    )
    replay.measurements = m
    with m.time("synthetic_replay_build_ms"):
        for block, offset in enumerate(
            range(0, replay_size, c.self_play.trajectory_length)
        ):
            replay.add(
                make_trajectory(
                    sampleable_steps=min(
                        c.self_play.trajectory_length, replay_size - offset
                    ),
                    lookahead_steps=max(c.training.unroll_steps, c.training.td_steps),
                    block_id=block,
                    stack_size=c.environment.frame_stack,
                    channels=channels,
                    screen_size=c.environment.screen_size,
                    action_space_size=actions,
                )
            )
    # Compile on the actual learner shape before any native worker is active.
    batch = replay.sample(
        c.training.batch_size, include_value_bootstraps=False, pin_memory=True
    ).to(device, keep_indices_on_cpu=True)
    with m.time("learner_compile_and_warmup_ms"):
        for _ in range(warmup):
            trainer.train_step(batch)
        torch.cuda.synchronize()
    del batch
    # Kernel warmup is not part of the simulated learner-update schedule.
    trainer._step_count = start_step
    pipeline = None
    try:
        if mode == "pipeline":
            pipeline = MeasuredPipeline(
                in_channels=c.environment.frame_stack * channels,
                action_space_size=actions,
                search_config=search,
                policy_chunk_size=c.reanalysis.policy_chunk_size,
                cache_targets=c.reanalysis.cache_targets,
                cache_target_ttl=c.reanalysis.cache_target_ttl,
                rng_seed=c.seed,
                support_min=-300,
                support_max=300,
                precision=c.training.precision,
                search_threads=c.reanalysis.worker_num_threads,
                prefetch_batches=c.reanalysis.prefetch_batches,
                timeout_seconds=c.reanalysis.timeout_seconds,
                target_update_interval=c.reanalysis.target_update_interval,
                device=device,
            )
            pipeline.measurements = m
            pipeline.publish_weights(
                start_step,
                make_target_state(
                    agent.representation_network,
                    agent.prediction_network,
                    agent.dynamics_network,
                ),
                wait=True,
            )
        with BatchWorker(
            replay,
            batch_size=c.training.batch_size,
            device=device,
            reanalysis_pipeline=pipeline,
            # Isolation uses stored TD targets: no MCTS metadata exists without reanalysis.
            value_target=c.training.value_target if pipeline is not None else "td",
            collection_steps=c.training.steps,
            mixed_value_start_step=c.training.mixed_value_start_step,
            mixed_value_threshold=c.training.mixed_value_threshold,
            priority_beta_initial=c.replay.priority_beta_initial,
            priority_beta_final=c.replay.priority_beta_final,
            priority_beta_steps=c.training.steps + c.training.final_steps,
            max_in_flight=c.training.batch_max_in_flight,
            ready_prefetch=c.training.batch_ready_prefetch,
            timeout_seconds=c.training.batch_worker_timeout_seconds,
        ) as worker:
            worker._replay_lock = TimedLock(m)
            step = start_step
            target_version = start_step

            def phase(count):
                nonlocal step, target_version
                events = []
                torch.cuda.synchronize()
                started = perf_counter()
                worker.start(step, count)
                for _ in range(count):
                    with m.time("learner_ready_wait_ms"):
                        ready = worker.next_ready()
                    a, b_event, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(3)
                    ]
                    a.record()
                    ready.wait_for_current_stream(device)
                    b_event.record()
                    with m.time("learner_enqueue_cpu_ms"):
                        metrics = trainer.train_step(ready.gpu_batch)
                    end.record()
                    with m.time("complete_cpu_ms"):
                        worker.complete(ready, metrics.priorities)
                    events.append((a, b_event, end))
                    for name in (
                        "sample_duration_ms",
                        "transfer_enqueue_ms",
                        "queue_wait_ms",
                        "worker_duration_ms",
                        "policy_roots_requested",
                        "policy_roots_searched",
                        "cache_hits",
                        "cache_target_age_mean",
                    ):
                        value = getattr(ready, name)
                        if value is not None:
                            m.values[name].append(value)
                    step += 1
                    if (
                        pipeline is not None
                        and step - target_version >= c.reanalysis.target_update_interval
                    ):
                        with m.time("target_snapshot_ms"):
                            state = make_target_state(
                                agent.representation_network,
                                agent.prediction_network,
                                agent.dynamics_network,
                            )
                        with m.time("target_publish_ms"):
                            worker.publish_weights(step, state)
                        target_version = step
                worker.wait_idle()
                torch.cuda.synchronize()
                elapsed = perf_counter() - started
                m.values["learner_gpu_stream_span_ms"].extend(
                    b.elapsed_time(end) for _, b, end in events
                )
                m.values["ready_event_gpu_wait_ms"].extend(
                    a.elapsed_time(b) for a, b, _ in events
                )
                return elapsed

            warm_seconds = phase(pipeline_warmup)
            setup = dict(m.values)
            results = []
            for repetition in range(repetitions):
                # Optional explicit cold-cache windows; otherwise retain cached targets.
                if pipeline is not None and cold_windows:
                    with m.time("boundary_cache_clear_ms"):
                        pipeline.clear_cache()
                boundary = m.values.get("boundary_cache_clear_ms", [])[-1:]
                m.values.clear()
                torch.cuda.reset_peak_memory_stats()
                if trace:
                    torch.cuda.cudart().cudaProfilerStart()
                elapsed = phase(updates)
                if trace:
                    torch.cuda.cudart().cudaProfilerStop()
                result = {
                    "repetition": repetition,
                    "start_step": step - updates,
                    "elapsed_seconds": elapsed,
                    "updates_per_second": updates / elapsed,
                    "boundary_cache_clear_ms": boundary,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "peak_memory_caveat": "Native requests reset device-wide allocator peaks; these are lower bounds, not reliable whole-window peaks."
                    if pipeline is not None
                    else None,
                    "distributions": {k: distribution(v) for k, v in m.values.items()},
                    "samples": dict(m.values),
                }
                results.append(result)
                (output / f"repeat_{repetition}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(
                    f"{mode} replay={replay_size} repeat={repetition}: {updates / elapsed:.3f} updates/s",
                    flush=True,
                )
            # Isolated publication includes snapshot and acknowledged publication; no queued batches.
            m.values.clear()
            if pipeline is not None:
                for publication in range(5):
                    with m.time("isolated_target_snapshot_ms"):
                        state = make_target_state(
                            agent.representation_network,
                            agent.prediction_network,
                            agent.dynamics_network,
                        )
                    with m.time("isolated_target_publication_ms"):
                        worker.publish_weights(step + publication + 1, state)
                    target_version = step + publication + 1
            state = make_target_state(
                agent.representation_network,
                agent.prediction_network,
                agent.dynamics_network,
            )
            with TemporaryDirectory(prefix="checkpoint-", dir=output) as temporary:
                for _ in range(3):
                    with m.time("training_checkpoint_pagecache_ms"):
                        save_checkpoint(
                            Path(temporary) / "checkpoint.pt",
                            agent=agent,
                            config=c,
                            trainer=trainer,
                            target_state=state,
                            target_version=target_version,
                            update=step,
                        )
            summary = {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "gpu": torch.cuda.get_device_name(),
                "torch_cpu_threads": torch.get_num_threads(),
                "action_space_size": actions,
                "mode": mode,
                "replay_size": replay_size,
                "synthetic_replay": True,
                "cold_windows": cold_windows,
                "pipeline_warmup_updates": pipeline_warmup,
                "benchmark_sha256": source_sha256,
                "warmup_pipeline_seconds": warm_seconds,
                "setup_samples": setup,
                "isolated_samples": dict(m.values),
                "throughput": distribution([r["updates_per_second"] for r in results]),
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    main()
