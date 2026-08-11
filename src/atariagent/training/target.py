"""Delayed target network for replay value and policy reanalysis."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
import math
import random
from typing import Literal

import torch
from torch import Tensor, nn

from atariagent.agent import BatchedNetworkEvaluator, categorical_to_scalar
from atariagent.replay import ReplayBatch
from atariagent.search import Evaluation, MCTS, MCTSConfig, SearchResult


Precision = Literal["fp32", "bf16"]


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
    ) -> None:
        super().__init__()
        if support_min >= support_max:
            raise ValueError("support_min must be less than support_max")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise TypeError("rng_seed must be an integer")
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
        self._selection_rng = random.Random(rng_seed)
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
        representation: nn.Module,
        prediction: nn.Module,
        dynamics: nn.Module | None = None,
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
    def reanalyze_batch(
        self,
        batch: ReplayBatch,
        *,
        reanalyze_values: bool,
        policy_ratio: float,
        policy_chunk_size: int,
        section: (
            Callable[[str], AbstractContextManager[None]] | None
        ) = None,
    ) -> ReplayBatch:
        """Apply configured value and policy refreshes to one replay batch."""
        if not isinstance(reanalyze_values, bool):
            raise TypeError("reanalyze_values must be a boolean")
        phase = section if section is not None else lambda _: nullcontext()
        with phase("target_reanalysis"):
            if reanalyze_values:
                batch = self.reanalyze_values(batch)
        with phase("policy_reanalysis"):
            if policy_ratio > 0.0:
                batch = self.reanalyze_policies(
                    batch,
                    ratio=policy_ratio,
                    chunk_size=policy_chunk_size,
                )
        return batch

    @torch.no_grad()
    def reanalyze_values(self, batch: ReplayBatch) -> ReplayBatch:
        """Replace stored MCTS value bootstraps with direct target predictions."""
        if batch.value_bootstrap_frames is None:
            # Compatibility for synthetic/legacy batches that already contain
            # complete scalar value targets.
            return batch
        if batch.value_bootstrap_mask is None:
            raise ValueError("batch has no value-bootstrap mask")

        fresh_values = torch.zeros_like(batch.value_targets)
        with self._autocast_context():
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
    def reanalyze_policies(
        self,
        batch: ReplayBatch,
        *,
        ratio: float,
        chunk_size: int,
    ) -> ReplayBatch:
        """Replace selected stored policies with fresh target-network MCTS."""
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("policy reanalysis ratio must be in [0, 1]")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("policy reanalysis chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("policy reanalysis chunk_size must be positive")

        reanalyze_count = math.floor(batch.batch_size * ratio)
        if reanalyze_count == 0:
            return batch
        if self._mcts is None or self._policy_evaluator is None:
            raise RuntimeError("target network has no policy MCTS components")

        selected_indices = self._selection_rng.sample(
            range(batch.batch_size), reanalyze_count
        )
        selected_mask = torch.zeros(
            batch.batch_size,
            dtype=torch.bool,
            device=batch.policy_targets.device,
        )
        selected_mask[selected_indices] = True
        positions = torch.nonzero(
            selected_mask[:, None] & batch.policy_mask,
            as_tuple=False,
        )
        if positions.shape[0] == 0:
            return batch

        fresh_policies = torch.zeros_like(batch.policy_targets)
        with self._autocast_context():
            for start in range(0, positions.shape[0], chunk_size):
                chunk = positions[start : start + chunk_size]
                observations = self._policy_observations(batch, chunk)
                results = self._search_policies(observations)
                policies = torch.as_tensor(
                    tuple(result.policy for result in results),
                    dtype=fresh_policies.dtype,
                    device=fresh_policies.device,
                )
                fresh_policies[chunk[:, 0], chunk[:, 1]] = policies

        return batch.with_reanalyzed_policy_targets(
            fresh_policies,
            selected_mask=selected_mask,
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
        frame_offsets = torch.arange(
            batch.stack_size,
            device=positions.device,
        )
        frames = batch.frames[
            positions[:, 0, None],
            positions[:, 1, None] + frame_offsets,
        ]
        channels, height, width = batch.frames.shape[2:]
        return frames.reshape(
            positions.shape[0],
            batch.stack_size * channels,
            height,
            width,
        ).to(dtype=torch.float32).div_(255.0)

    def _search_policies(
        self,
        observations: Tensor,
    ) -> tuple[SearchResult, ...]:
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
        policy_rows = policy_logits.float().cpu().tolist()
        scalar_values = values.float().cpu().tolist()
        roots = tuple(
            Evaluation(
                state=states[index],
                value_prefix=0.0,
                value=scalar_values[index],
                policy_logits=policy_rows[index],
            )
            for index in range(states.shape[0])
        )
        # EfficientZero V1 reanalysis uses normalized raw visit counts. The
        # sampled MCTS action is intentionally ignored.
        return self._mcts.search_batch(
            roots,
            self._policy_evaluator,
            add_exploration_noise=True,
            temperature=1.0,
        )

    def _autocast_context(self):
        parameter = next(self.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        if self.precision == "bf16":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return nullcontext()


__all__ = ["ValueTargetNetwork"]
