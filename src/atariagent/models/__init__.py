"""Neural-network components used by AtariAgent."""

from .dynamics import DynamicsNetwork, RewardPredictionNetwork
from .representation import RepresentationNetwork, ResidualBlock

__all__ = [
    "DynamicsNetwork",
    "RepresentationNetwork",
    "ResidualBlock",
    "RewardPredictionNetwork",
]
