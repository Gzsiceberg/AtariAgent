"""Delayed target network for replay value and policy reanalysis."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import random
from typing import Literal, Protocol

import torch
from torch import Tensor, nn

from atariagent.agent import BatchedNetworkEvaluator, categorical_to_scalar
from atariagent.replay_batch import ReplayBatch
from atariagent.search import MCTS, MCTSConfig


Precision = Literal["fp32", "bf16"]


class StateSource(Protocol):
    def state_dict(self) -> dict[str, Tensor]: ...


class ValueTargetNetwork(nn.Module):
    """Frozen target snapshot used for direct values and optional policy MCTS."""

    def __init__(
        self,
        representation: nn.Module,
        prediction: nn.Module,
        *,
        dynamics: nn.Module | None = None,
        action_space_size: int | None = None,
        mcts_config: MCTSConfig | None = None,
        rng_seed: int = 0,
        support_min: int = -300,
        support_max: int = 300,
        precision: Precision = "fp32",
        chunk_size: int = 768,
    ) -> None:
        super().__init__()
        if support_min >= support_max:
            raise ValueError("support_min must be less than support_max")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise TypeError("rng_seed must be an integer")
        self._validate_chunk_size(chunk_size)
        if dynamics is not None and (
            action_space_size is None or action_space_size <= 0
        ):
            raise ValueError(
                "action_space_size must be positive for policy reanalysis"
            )

        self.representation = deepcopy(representation)
        self.dynamics = deepcopy(dynamics) if dynamics is not None else None
        self.prediction = deepcopy(prediction)
        self.support_min = support_min
        self.support_max = support_max
        self.precision: Precision = precision
        self.chunk_size = chunk_size
        self._mcts: MCTS | None = None
        self._policy_evaluator: BatchedNetworkEvaluator | None = None
        if self.dynamics is not None:
            assert action_space_size is not None
            def scalar_decoder(logits: Tensor) -> Tensor:
                return categorical_to_scalar(
                    logits.float(),
                    support_min=support_min,
                    support_max=support_max,
                )

            self._policy_evaluator = BatchedNetworkEvaluator(
                self.dynamics,
                self.prediction,
                action_space_size=action_space_size,
                value_decoder=scalar_decoder,
                value_prefix_decoder=scalar_decoder,
            )
            self._mcts = MCTS(
                mcts_config,
                evaluator=self._policy_evaluator,
                rng=random.Random(rng_seed + 1),
            )

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> ValueTargetNetwork:
        """Keep target batch-normalization statistics frozen in eval mode."""
        return super().train(False)

    @torch.no_grad()
    def synchronize(
        self,
        representation: nn.Module | StateSource,
        prediction: nn.Module | StateSource,
        dynamics: nn.Module | StateSource | None = None,
    ) -> None:
        """Hard-copy online parameters and batch-normalization buffers."""
        self.representation.load_state_dict(representation.state_dict())
        self.prediction.load_state_dict(prediction.state_dict())
        if self.dynamics is not None:
            if dynamics is None:
                raise ValueError("dynamics is required for policy reanalysis")
            self.dynamics.load_state_dict(dynamics.state_dict())
        self.eval()

    @torch.no_grad()
    def reanalyze_batch(self, batch: ReplayBatch) -> ReplayBatch:
        """Refresh value and policy targets for one replay batch."""
        batch = self.reanalyze_values(batch)
        return self.reanalyze_policies(batch)

    @torch.no_grad()
    def reanalyze_values(self, batch: ReplayBatch) -> ReplayBatch:
        """Replace stored MCTS value bootstraps with direct target predictions."""
        if batch.value_bootstrap_frames is None:
            # Compatibility for synthetic/legacy batches that already contain
            # complete scalar value targets.
            return batch
        if batch.value_bootstrap_mask is None:
            raise ValueError("batch has no value-bootstrap mask")
        if batch.value_bootstrap_values is None:
            raise ValueError("batch has no stored bootstrap values")
        positions = torch.nonzero(
            batch.value_bootstrap_mask,
            as_tuple=False,
        )
        if positions.shape[0] == 0:
            return batch

        fresh_values = batch.value_bootstrap_values.clone()
        with self._autocast_context():
            for start in range(0, positions.shape[0], self.chunk_size):
                chunk = positions[start : start + self.chunk_size]
                observations = self._stacked_observations(
                    batch.value_bootstrap_frames,
                    chunk,
                    stack_size=batch.stack_size,
                )
                state = self.representation(observations)
                _, value_logits = self.prediction(state)
                values = categorical_to_scalar(
                    value_logits.float(),
                    support_min=self.support_min,
                    support_max=self.support_max,
                )
                fresh_values[chunk[:, 0], chunk[:, 1]] = values
        return batch.with_reanalyzed_value_targets(fresh_values)

    @torch.no_grad()
    def reanalyze_policies(self, batch: ReplayBatch) -> ReplayBatch:
        """Replace selected stored policies with fresh target-network MCTS."""
        if self._mcts is None or self._policy_evaluator is None:
            raise RuntimeError("target network has no policy MCTS components")

        positions = torch.nonzero(batch.policy_mask, as_tuple=False)
        if positions.shape[0] == 0:
            return batch

        fresh_policies = batch.policy_targets.clone()
        with self._autocast_context():
            for start in range(0, positions.shape[0], self.chunk_size):
                chunk = positions[start : start + self.chunk_size]
                observations = self._policy_observations(batch, chunk)
                policies = self._search_policies(observations)
                fresh_policies[chunk[:, 0], chunk[:, 1]] = policies.to(
                    dtype=fresh_policies.dtype,
                )

        # Every changed position came directly from policy_mask; the cloned
        # tensor already preserves all unselected stored targets.
        return batch.with_reanalysis_targets(
            value_targets=batch.value_targets,
            policy_targets=fresh_policies,
        )

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

    def _policy_observations(
        self,
        batch: ReplayBatch,
        positions: Tensor,
    ) -> Tensor:
        return self._stacked_observations(
            batch.frames,
            positions,
            stack_size=batch.stack_size,
        )

    @staticmethod
    def _stacked_observations(
        frames: Tensor,
        positions: Tensor,
        *,
        stack_size: int,
    ) -> Tensor:
        """Gather and normalize arbitrary overlapping frame stacks."""
        frame_offsets = torch.arange(
            stack_size,
            device=positions.device,
        )
        selected = frames[
            positions[:, 0, None],
            positions[:, 1, None] + frame_offsets,
        ]
        channels, height, width = frames.shape[2:]
        return selected.reshape(
            positions.shape[0],
            stack_size * channels,
            height,
            width,
        ).to(dtype=torch.float32).div_(255.0)

    def _search_policies(
        self,
        observations: Tensor,
    ) -> Tensor:
        """Run packed native MCTS and return normalized visit policies."""
        assert self._mcts is not None
        assert self._policy_evaluator is not None
        states = self.representation(observations)
        policy_logits, value_logits = self.prediction(states)
        self._policy_evaluator.validate_policy(policy_logits, states.shape[0])
        values = self._policy_evaluator.decode(
            self._policy_evaluator.value_decoder,
            value_logits.float(),
            "value_decoder",
        )
        results = self._mcts.search_batch(
            states,
            values,
            policy_logits,
            add_exploration_noise=True,
        )
        visits = torch.from_numpy(results.visit_counts).to(
            dtype=policy_logits.dtype,
            device=policy_logits.device,
        )
        return visits / visits.sum(dim=1, keepdim=True)

    @staticmethod
    def _validate_chunk_size(chunk_size: int) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def _autocast_context(self):
        parameter = next(self.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        if self.precision == "bf16":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return nullcontext()


__all__ = ["ValueTargetNetwork"]
