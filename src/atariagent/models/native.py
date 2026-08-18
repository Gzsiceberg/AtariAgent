"""Inference-only LibTorch model implementations.

The training models in :mod:`atariagent.models` remain Python ``nn.Module``
classes.  This module exposes matching C++ modules whose parameter and buffer names
intentionally match the Python implementations exactly.
"""

from dataclasses import dataclass

import torch

from atariagent.search import SearchConfig

from ._models_native import (
    BatchedNetworkEvaluator,
    DynamicsNetwork,
    NativeReanalysisEngine,
    PolicyNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    ResidualBlock,
    RewardPredictionNetwork,
    TreeSearch,
    ValueNetwork,
    ValueTargetNetwork,
    categorical_to_scalar,
    set_tree_search_num_threads,
)

# Compatibility aliases for callers using the former native names.
MCTS = TreeSearch
set_mcts_num_threads = set_tree_search_num_threads


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


def make_value_target(
    models: InferenceModels,
    action_space_size: int,
    config: SearchConfig | None = None,
    *,
    seed: int = 0,
    support_min: int = -300,
    support_max: int = 300,
    chunk_size: int = 768,
    precision: str = "fp32",
) -> ValueTargetNetwork:
    """Compose native models into the fully C++ target-reanalysis stack."""
    config = config or SearchConfig()
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
        config.search_algorithm,
        config.num_top_actions,
        config.c_visit,
        config.c_scale,
    )


__all__ = [
    "MCTS",
    "BatchedNetworkEvaluator",
    "DynamicsNetwork",
    "InferenceModels",
    "NativeReanalysisEngine",
    "PolicyNetwork",
    "PredictionNetwork",
    "RepresentationNetwork",
    "ResidualBlock",
    "RewardPredictionNetwork",
    "TreeSearch",
    "ValueNetwork",
    "ValueTargetNetwork",
    "categorical_to_scalar",
    "make_value_target",
    "set_mcts_num_threads",
    "set_tree_search_num_threads",
]
