#!/usr/bin/env python3
"""Time real collection and the unchanged full evaluation protocol in isolation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import hydra
import torch
from atariagent.search._tree_search_native import set_num_threads
from benchmark_training_performance import distribution
from omegaconf import DictConfig, OmegaConf
from train_agent import (
    configure_training_backend,
    create_environments,
    create_evaluation_environment,
)

from atariagent.agent import AtariAgent
from atariagent.evaluation import evaluate_agent
from atariagent.search import SearchConfig
from atariagent.selfplay import SelfPlayWorker
from atariagent.typecheck import set_runtime_typechecking


@hydra.main(version_base=None, config_path="../configs", config_name="train_agent")
def main(c: DictConfig):
    b = c.get("benchmark", {})
    output = Path(b.get("output", "measurements/actor"))
    output.mkdir(parents=True, exist_ok=True)
    repetitions = int(b.get("repetitions", 3))
    evaluation_repetitions = int(b.get("evaluation_repetitions", 1))
    if repetitions < 1 or evaluation_repetitions < 1:
        raise ValueError("repetitions must be positive")
    OmegaConf.save(c, output / "resolved_config.yaml", resolve=True)
    set_runtime_typechecking(c.training.runtime_type_checks)
    configure_training_backend(
        c.training.deterministic, cudnn_benchmark=c.training.cudnn_benchmark
    )
    set_num_threads(c.reanalysis.worker_num_threads)
    torch.manual_seed(c.seed)
    envs = create_environments(c)
    channels = 1 if c.environment.grayscale else 3
    agent = (
        AtariAgent(
            c.environment.frame_stack * channels,
            int(envs[0].action_space.n),
            search_config=SearchConfig(
                num_simulations=c.self_play.num_simulations,
                discount=c.training.discount**c.environment.frame_skip,
                value_prefix_horizon=c.training.lstm_horizon,
                search_algorithm=c.self_play.search_algorithm,
                root_exploration_fraction=c.self_play.root_exploration_fraction,
                c_visit=c.self_play.c_visit,
                c_scale=c.self_play.c_scale,
            ),
            search_rng=random.Random(c.seed),
        )
        .to("cuda")
        .eval()
    )
    checkpoint = b.get("checkpoint", None)
    checkpoint_hash = None
    if checkpoint:
        checkpoint_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for name in ("representation", "dynamics", "prediction"):
            getattr(agent, f"{name}_network").load_state_dict(state[name])
        del state
    worker = SelfPlayWorker(
        agent,
        environments=envs,
        frame_stack=c.environment.frame_stack,
        trajectory_length=c.self_play.trajectory_length,
        lookahead_steps=max(c.training.unroll_steps, c.training.td_steps),
        base_seed=c.seed,
        clip_rewards=c.self_play.clip_rewards,
    )
    collection = []
    try:
        start = perf_counter()
        worker.run(c.self_play.steps_per_iteration)
        torch.cuda.synchronize()
        warmup_seconds = perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repetitions):
            start = perf_counter()
            worker.run(c.self_play.steps_per_iteration)
            torch.cuda.synchronize()
            collection.append(perf_counter() - start)
    finally:
        worker.close()
    result = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "weights": "checkpoint" if checkpoint else "random_initialization",
        "collection_warmup_seconds": warmup_seconds,
        "collection_seconds": collection,
        "collection_transitions_per_window": c.self_play.num_envs
        * c.self_play.steps_per_iteration,
        "collection_distribution_seconds": distribution(collection),
        "collection_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "evaluation": [],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for repetition in range(evaluation_repetitions):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        stats = evaluate_agent(
            agent,
            lambda: create_evaluation_environment(c),
            episodes=c.evaluation.episodes,
            num_envs=c.evaluation.num_envs,
            seed=c.seed,
        )
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        result["evaluation"].append(
            {
                "seconds": elapsed,
                "rewards": stats.rewards,
                "mean_score": stats.mean,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"evaluation repeat={repetition}: {elapsed:.3f}s, {c.evaluation.episodes} episodes",
            flush=True,
        )


if __name__ == "__main__":
    main()
