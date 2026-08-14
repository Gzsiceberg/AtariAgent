"""Inference-only LibTorch model implementations.

The training models in :mod:`atariagent.models` remain Python ``nn.Module``
classes.  This module exposes matching C++ modules and a loader for training
checkpoints; their parameter and buffer names intentionally match the Python
implementations exactly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from atariagent.search import MCTSConfig

from ._models_native import (
    MCTS,
    BatchedNetworkEvaluator,
    DynamicsNetwork,
    PolicyNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    ResidualBlock,
    RewardPredictionNetwork,
    ValueNetwork,
    ValueTargetNetwork,
    categorical_to_scalar,
)


@dataclass(frozen=True)
class InferenceModels:
    """The three checkpointed networks used by MuZero inference."""

    representation: RepresentationNetwork
    dynamics: DynamicsNetwork
    prediction: PredictionNetwork

    def eval(self) -> "InferenceModels":
        """Put every network into inference mode and return this bundle."""
        self.representation.eval()
        self.dynamics.eval()
        self.prediction.eval()
        return self

    def to(self, device: torch.device | str) -> "InferenceModels":
        """Move every network to ``device`` and return this bundle."""
        resolved_device = str(torch.device(device))
        self.representation.to(resolved_device)
        self.dynamics.to(resolved_device)
        self.prediction.to(resolved_device)
        return self


def make_mcts(
    models: InferenceModels,
    action_space_size: int,
    config: MCTSConfig | None = None,
    *,
    seed: int = 0,
    support_min: int = -300,
    support_max: int = 300,
) -> MCTS:
    """Compose native models into the fully C++ evaluator and MCTS loop."""
    config = config or MCTSConfig()
    evaluator = BatchedNetworkEvaluator(
        models.dynamics,
        models.prediction,
        action_space_size,
        support_min,
        support_max,
    )
    return MCTS(
        evaluator,
        config.num_simulations,
        config.discount,
        config.pb_c_init,
        config.pb_c_base,
        config.value_delta_max,
        config.dirichlet_alpha,
        config.root_exploration_fraction,
        config.value_prefix_horizon,
        seed,
    )


def make_value_target(
    models: InferenceModels,
    action_space_size: int,
    config: MCTSConfig | None = None,
    *,
    seed: int = 0,
    support_min: int = -300,
    support_max: int = 300,
    chunk_size: int = 768,
    precision: str = "fp32",
) -> ValueTargetNetwork:
    """Compose native models into the fully C++ target-reanalysis stack."""
    config = config or MCTSConfig()
    return ValueTargetNetwork(
        models.representation,
        models.prediction,
        models.dynamics,
        action_space_size,
        support_min,
        support_max,
        chunk_size,
        precision,
        config.num_simulations,
        config.discount,
        config.pb_c_init,
        config.pb_c_base,
        config.value_delta_max,
        config.dirichlet_alpha,
        config.root_exploration_fraction,
        config.value_prefix_horizon,
        seed,
    )


def load_inference_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[InferenceModels, dict[str, Any]]:
    """Load a Python training checkpoint into inference-only C++ modules.

    Architecture dimensions are inferred from checkpoint tensors, avoiding
    environment-specific assumptions about frame stacking or action count.
    The returned checkpoint mapping is useful for reading its saved config.
    """
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    required = ("representation", "dynamics", "prediction")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint is missing model state: {', '.join(missing)}")

    representation_state = checkpoint["representation"]
    prediction_state = checkpoint["prediction"]
    in_channels = int(representation_state["stem.0.weight"].shape[1])
    action_space_size = int(prediction_state["policy.projection.3.weight"].shape[0])
    value_support_size = int(prediction_state["value.projection.3.weight"].shape[0])

    models = InferenceModels(
        representation=RepresentationNetwork(in_channels),
        dynamics=DynamicsNetwork(action_space_size),
        prediction=PredictionNetwork(action_space_size, value_support_size),
    )
    models.representation.load_state_dict(representation_state)
    models.dynamics.load_state_dict(checkpoint["dynamics"])
    models.prediction.load_state_dict(prediction_state)
    models.eval().to(device)
    return models, checkpoint


__all__ = [
    "MCTS",
    "BatchedNetworkEvaluator",
    "DynamicsNetwork",
    "InferenceModels",
    "PolicyNetwork",
    "PredictionNetwork",
    "RepresentationNetwork",
    "ResidualBlock",
    "RewardPredictionNetwork",
    "ValueNetwork",
    "ValueTargetNetwork",
    "categorical_to_scalar",
    "load_inference_checkpoint",
    "make_mcts",
    "make_value_target",
]
