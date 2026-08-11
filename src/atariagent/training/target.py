"""Delayed target network for direct replay value reanalysis."""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from typing import Literal

import torch
from torch import Tensor, nn

from atariagent.agent import categorical_to_scalar
from atariagent.replay import ReplayBatch


Precision = Literal["fp32", "bf16"]


class ValueTargetNetwork(nn.Module):
    """Hard-copied representation/value network used for n-step bootstraps.

    The target evaluates real replay observations with initial inference only.
    It intentionally performs neither MCTS nor recurrent dynamics unrolling.
    """

    def __init__(
        self,
        representation: nn.Module,
        prediction: nn.Module,
        *,
        support_min: int = -300,
        support_max: int = 300,
        precision: Precision = "fp32",
    ) -> None:
        super().__init__()
        if support_min >= support_max:
            raise ValueError("support_min must be less than support_max")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")

        self.representation = deepcopy(representation)
        self.prediction = deepcopy(prediction)
        self.support_min = support_min
        self.support_max = support_max
        self.precision: Precision = precision
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> ValueTargetNetwork:
        """Keep target batch-normalization statistics frozen in eval mode."""
        return super().train(False)

    @torch.no_grad()
    def synchronize(
        self,
        representation: nn.Module,
        prediction: nn.Module,
    ) -> None:
        """Hard-copy current online parameters and batch-normalization buffers."""
        self.representation.load_state_dict(representation.state_dict())
        self.prediction.load_state_dict(prediction.state_dict())
        self.eval()

    @torch.no_grad()
    def reanalyze(self, batch: ReplayBatch) -> ReplayBatch:
        """Replace stored MCTS value bootstraps with direct target predictions."""
        if batch.value_bootstrap_frames is None:
            # Compatibility for synthetic/legacy batches that already contain
            # complete scalar value targets.
            return batch
        if batch.value_bootstrap_mask is None:
            raise ValueError("batch has no value-bootstrap mask")

        fresh_values = torch.zeros_like(batch.value_targets)
        parameter = next(self.parameters(), None)
        device = parameter.device if parameter is not None else batch.frames.device
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        with autocast_context:
            for offset in range(batch.unroll_steps + 1):
                observations = batch.normalized_value_bootstrap_observation(
                    offset
                )
                state = self.representation(observations)
                _, value_logits = self.prediction(state)
                fresh_values[:, offset] = categorical_to_scalar(
                    value_logits.float(),
                    support_min=self.support_min,
                    support_max=self.support_max,
                )
        return batch.with_reanalyzed_value_targets(fresh_values)

    @torch.no_grad()
    def decoded_values(self, observations: Tensor) -> Tensor:
        """Return direct scalar values for diagnostic and unit-test inference."""
        state = self.representation(observations)
        _, value_logits = self.prediction(state)
        return categorical_to_scalar(
            value_logits.float(),
            support_min=self.support_min,
            support_max=self.support_max,
        )


__all__ = ["ValueTargetNetwork"]
