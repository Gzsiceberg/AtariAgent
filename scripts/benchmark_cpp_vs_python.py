#!/usr/bin/env python3
"""Benchmark every native inference component against its Python reference."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

import atariagent.models as python_models
from atariagent.agent import (
    BatchedNetworkEvaluator as PythonBatchedNetworkEvaluator,
)
from atariagent.agent import categorical_to_scalar as python_categorical_to_scalar
from atariagent.models import native
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTS as PythonMCTS
from atariagent.search import MCTSConfig
from atariagent.training import ValueTargetNetwork as PythonValueTargetNetwork
from atariagent.training.muzero_config import (
    EnvironmentConfig,
    checkpoint_path_for_environment,
)
from atariagent.typecheck import set_runtime_typechecking

Operation = Callable[[], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        default=EnvironmentConfig().id,
        help="environment ID used to derive the default checkpoint path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint path (default: derived from --environment)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--search-roots", type=int, default=32)
    parser.add_argument("--target-batch-size", type=int, default=8)
    parser.add_argument("--unroll-steps", type=int, default=2)
    parser.add_argument("--num-simulations", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--model-warmup", type=int, default=5)
    parser.add_argument("--model-iterations", type=int, default=20)
    parser.add_argument("--search-warmup", type=int, default=1)
    parser.add_argument("--search-iterations", type=int, default=5)
    parser.add_argument("--target-warmup", type=int, default=1)
    parser.add_argument("--target-iterations", type=int, default=3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/cpp_vs_python_all.json"),
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    operation: Operation,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    for _ in range(warmup):
        result = operation()
        del result
    synchronize(device)
    baseline_memory = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)

    durations: list[float] = []
    for _ in range(iterations):
        synchronize(device)
        started = perf_counter()
        result = operation()
        synchronize(device)
        durations.append((perf_counter() - started) * 1_000.0)
        del result
    peak_memory = 0
    if device.type == "cuda":
        peak_memory = max(
            0,
            torch.cuda.max_memory_allocated(device) - baseline_memory,
        )
    return {
        "durations_ms": durations,
        "mean_ms": mean(durations),
        "median_ms": median(durations),
        "peak_incremental_memory_bytes": peak_memory,
    }


def benchmark_pair(
    python_operation: Operation,
    cpp_operation: Operation,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    python_result = benchmark(
        python_operation,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    cpp_result = benchmark(
        cpp_operation,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    python_median = float(python_result["median_ms"])
    cpp_median = float(cpp_result["median_ms"])
    return {
        "python": python_result,
        "cpp": cpp_result,
        "speedup": python_median / cpp_median,
    }


def load_python_models(
    checkpoint: dict[str, Any],
    *,
    in_channels: int,
    action_space_size: int,
    value_support_size: int,
    device: torch.device,
):
    representation = python_models.RepresentationNetwork(in_channels)
    dynamics = python_models.DynamicsNetwork(action_space_size)
    prediction = python_models.PredictionNetwork(
        action_space_size,
        value_support_size=value_support_size,
    )
    representation.load_state_dict(checkpoint["representation"])
    dynamics.load_state_dict(checkpoint["dynamics"])
    prediction.load_state_dict(checkpoint["prediction"])
    return (
        representation.eval().to(device),
        dynamics.eval().to(device),
        prediction.eval().to(device),
    )


def make_target_batch(
    *,
    batch_size: int,
    unroll_steps: int,
    action_space_size: int,
    device: torch.device,
) -> ReplayBatch:
    states = unroll_steps + 1
    frame_count = 4 + unroll_steps
    frames = torch.randint(
        0,
        256,
        (batch_size, frame_count, 3, 96, 96),
        dtype=torch.uint8,
        device=device,
    )
    return ReplayBatch(
        frames=frames,
        actions=torch.zeros(
            batch_size, unroll_steps, 1, dtype=torch.long, device=device
        ),
        rewards=torch.zeros(batch_size, unroll_steps, device=device),
        policy_targets=torch.full(
            (batch_size, states, action_space_size),
            1.0 / action_space_size,
            device=device,
        ),
        value_targets=torch.zeros(batch_size, states, device=device),
        action_mask=torch.ones(
            batch_size, unroll_steps, dtype=torch.bool, device=device
        ),
        policy_mask=torch.ones(batch_size, states, dtype=torch.bool, device=device),
        value_mask=torch.ones(batch_size, states, dtype=torch.bool, device=device),
        indices=torch.arange(batch_size, device=device),
        importance_weights=torch.ones(batch_size, device=device),
        value_bootstrap_frames=frames.clone(),
        value_bootstrap_values=torch.zeros(batch_size, states, device=device),
        value_bootstrap_discounts=torch.ones(batch_size, states, device=device),
        value_bootstrap_mask=torch.ones(
            batch_size, states, dtype=torch.bool, device=device
        ),
    )


def native_module(cpp_type, python_module, device: torch.device, *args):
    module = cpp_type(*args)
    module.load_state_dict(python_module.state_dict())
    module.eval().to(str(device))
    return module


def assert_nested_close(actual, expected) -> None:
    if isinstance(actual, Tensor):
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_nested_close(actual_item, expected_item)
        return
    raise TypeError(f"unsupported correctness output: {type(actual)!r}")


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = Path(
            checkpoint_path_for_environment(args.environment)
        )
    sizes = (
        args.model_batch_size,
        args.search_roots,
        args.target_batch_size,
        args.unroll_steps,
        args.num_simulations,
        args.chunk_size,
        args.model_iterations,
        args.search_iterations,
        args.target_iterations,
    )
    if any(value <= 0 for value in sizes):
        raise ValueError("benchmark sizes and iterations must be positive")
    if (
        min(
            args.model_warmup,
            args.search_warmup,
            args.target_warmup,
        )
        < 0
    ):
        raise ValueError("warmup counts must be non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    set_runtime_typechecking(False)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    native_models, checkpoint = native.load_inference_checkpoint(
        args.checkpoint,
        device=device,
    )
    in_channels = int(checkpoint["representation"]["stem.0.weight"].shape[1])
    action_space_size = int(
        checkpoint["prediction"]["policy.projection.3.weight"].shape[0]
    )
    value_support_size = int(
        checkpoint["prediction"]["value.projection.3.weight"].shape[0]
    )
    representation, dynamics, prediction = load_python_models(
        checkpoint,
        in_channels=in_channels,
        action_space_size=action_space_size,
        value_support_size=value_support_size,
        device=device,
    )

    residual = python_models.ResidualBlock(64).eval().to(device)
    native_residual = native_module(native.ResidualBlock, residual, device, 64)
    reward = python_models.RewardPredictionNetwork().eval().to(device)
    native_reward = native_module(native.RewardPredictionNetwork, reward, device)
    policy = python_models.PolicyNetwork(action_space_size).eval().to(device)
    native_policy = native_module(
        native.PolicyNetwork, policy, device, action_space_size
    )
    value = python_models.ValueNetwork().eval().to(device)
    native_value = native_module(native.ValueNetwork, value, device)

    model_batch = args.model_batch_size
    observations = torch.rand(model_batch, in_channels, 96, 96, device=device)
    states = torch.rand(model_batch, 64, 6, 6, device=device)
    actions = torch.randint(0, action_space_size, (model_batch, 1), device=device)
    hidden = reward.initial_hidden(model_batch, device=device)
    support_logits = torch.randn(model_batch, 601, device=device)

    python_evaluator = PythonBatchedNetworkEvaluator(
        dynamics,
        prediction,
        action_space_size=action_space_size,
        value_decoder=python_categorical_to_scalar,
        value_prefix_decoder=python_categorical_to_scalar,
    )
    cpp_evaluator = native.BatchedNetworkEvaluator(
        native_models.dynamics,
        native_models.prediction,
        action_space_size,
    )
    reset = torch.zeros(model_batch, dtype=torch.bool, device=device)
    reset[::2] = True

    operations: dict[str, dict[str, object]] = {}
    model_pairs: dict[str, tuple[Operation, Operation]] = {
        "residual_block": (
            lambda: residual(states),
            lambda: native_residual(states),
        ),
        "representation": (
            lambda: representation(observations),
            lambda: native_models.representation(observations),
        ),
        "reward_prediction": (
            lambda: reward(states, hidden),
            lambda: native_reward(states, hidden),
        ),
        "dynamics": (
            lambda: dynamics(states, actions, hidden),
            lambda: native_models.dynamics(states, actions, hidden),
        ),
        "policy": (
            lambda: policy(states),
            lambda: native_policy(states),
        ),
        "value": (
            lambda: value(states),
            lambda: native_value(states),
        ),
        "prediction": (
            lambda: prediction(states),
            lambda: native_models.prediction(states),
        ),
        "categorical_to_scalar": (
            lambda: python_categorical_to_scalar(support_logits),
            lambda: native.categorical_to_scalar(support_logits),
        ),
        "batched_network_evaluator": (
            lambda: python_evaluator.evaluate_tensors(states, actions, hidden, reset),
            lambda: cpp_evaluator.evaluate_tensors(states, actions, hidden, reset),
        ),
    }

    with torch.inference_mode():
        for name, (python_operation, cpp_operation) in model_pairs.items():
            assert_nested_close(cpp_operation(), python_operation())
            operations[name] = benchmark_pair(
                python_operation,
                cpp_operation,
                warmup=args.model_warmup,
                iterations=args.model_iterations,
                device=device,
            )

        config = MCTSConfig(
            num_simulations=args.num_simulations,
            value_prefix_horizon=5,
        )
        search_states = states[: args.search_roots]
        search_policy_logits, search_value_logits = prediction(search_states)
        search_values = python_categorical_to_scalar(search_value_logits)
        python_mcts = PythonMCTS(
            config,
            evaluator=python_evaluator,
            rng=random.Random(0),
        )
        cpp_mcts = native.make_mcts(
            native_models,
            action_space_size,
            config,
            seed=0,
        )

        def python_search():
            return python_mcts.search_batch(
                search_states,
                search_values,
                search_policy_logits,
                _deterministic_ties=True,
            )

        def cpp_search():
            return cpp_mcts.search_batch(
                search_states,
                search_values,
                search_policy_logits,
                False,
                True,
            )

        expected_search = python_search()
        cpp_visits, cpp_root_values = cpp_search()
        np.testing.assert_array_equal(cpp_visits.numpy(), expected_search.visit_counts)
        np.testing.assert_allclose(
            cpp_root_values.numpy(),
            expected_search.root_values,
            rtol=1e-6,
            atol=1e-6,
        )
        operations["mcts"] = benchmark_pair(
            python_search,
            cpp_search,
            warmup=args.search_warmup,
            iterations=args.search_iterations,
            device=device,
        )

        target_batch = make_target_batch(
            batch_size=args.target_batch_size,
            unroll_steps=args.unroll_steps,
            action_space_size=action_space_size,
            device=device,
        )
        python_target = PythonValueTargetNetwork(
            representation,
            prediction,
            dynamics=dynamics,
            action_space_size=action_space_size,
            mcts_config=config,
            rng_seed=0,
            precision=args.precision,
            chunk_size=args.chunk_size,
        ).to(device)
        cpp_target = native.make_value_target(
            native_models,
            action_space_size,
            config,
            seed=0,
            precision=args.precision,
            chunk_size=args.chunk_size,
        )

        def python_values():
            return python_target.reanalyze_values(target_batch).value_targets

        def cpp_values():
            assert target_batch.value_bootstrap_frames is not None
            assert target_batch.value_bootstrap_mask is not None
            assert target_batch.value_bootstrap_values is not None
            assert target_batch.value_bootstrap_discounts is not None
            return cpp_target.reanalyze_values(
                target_batch.value_bootstrap_frames,
                target_batch.value_bootstrap_mask,
                target_batch.value_bootstrap_values,
                target_batch.value_bootstrap_discounts,
                target_batch.value_targets,
                target_batch.stack_size,
            )

        torch.testing.assert_close(cpp_values(), python_values())
        operations["value_target_values"] = benchmark_pair(
            python_values,
            cpp_values,
            warmup=args.target_warmup,
            iterations=args.target_iterations,
            device=device,
        )
        operations["value_target_policies"] = benchmark_pair(
            lambda: python_target.reanalyze_policies(target_batch),
            lambda: cpp_target.reanalyze_policies(
                target_batch.frames,
                target_batch.policy_mask,
                target_batch.policy_targets,
                target_batch.stack_size,
            ),
            warmup=args.target_warmup,
            iterations=args.target_iterations,
            device=device,
        )
        operations["value_target_full_batch"] = benchmark_pair(
            lambda: python_target.reanalyze_batch(target_batch),
            lambda: cpp_target.reanalyze_batch(target_batch),
            warmup=args.target_warmup,
            iterations=args.target_iterations,
            device=device,
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "model_batch_size": args.model_batch_size,
        "search_roots": args.search_roots,
        "target_batch_size": args.target_batch_size,
        "unroll_steps": args.unroll_steps,
        "num_simulations": args.num_simulations,
        "chunk_size": args.chunk_size,
        "precision": args.precision,
        "operations": operations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
