"""Delayed target network for replay value and policy reanalysis."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import random
from typing import Literal

import torch
from torch import Tensor, nn

from atariagent.agent import BatchedNetworkEvaluator, categorical_to_scalar
from atariagent.replay_batch import ReplayBatch
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
        policy_chunk_size: int,
    ) -> ReplayBatch:
        """Refresh value and policy targets for one replay batch."""
        batch = self.reanalyze_values(batch)
        return self.reanalyze_policies(
            batch,
            chunk_size=policy_chunk_size,
        )

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
        if not batch.value_bootstrap_mask.any():
            return batch

        fresh_values = batch.value_bootstrap_values.clone()
        with self._autocast_context():
            for offset in range(batch.unroll_steps + 1):
                rows = batch.value_bootstrap_mask[:, offset].nonzero().flatten()
                if rows.numel() == 0:
                    continue
                observations = batch.normalized_value_bootstrap_observation(
                    offset
                )[rows]
                state = self.representation(observations)
                _, value_logits = self.prediction(state)
                fresh_values[rows, offset] = categorical_to_scalar(
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
        chunk_size: int,
    ) -> ReplayBatch:
        """Replace selected stored policies with fresh target-network MCTS."""
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("policy reanalysis chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("policy reanalysis chunk_size must be positive")

        if self._mcts is None or self._policy_evaluator is None:
            raise RuntimeError("target network has no policy MCTS components")

        positions = torch.nonzero(batch.policy_mask, as_tuple=False)
        if positions.shape[0] == 0:
            return batch

        fresh_policies = batch.policy_targets.clone()
        with self._autocast_context():
            for start in range(0, positions.shape[0], chunk_size):
                chunk = positions[start : start + chunk_size]
                observations = self._policy_observations(batch, chunk)
                results = self._search_policies(observations)
                visits = torch.as_tensor(
                    tuple(result.visit_counts for result in results),
                    dtype=fresh_policies.dtype,
                    device=fresh_policies.device,
                )
                policies = visits / visits.sum(dim=1, keepdim=True)
                fresh_policies[chunk[:, 0], chunk[:, 1]] = policies

        return batch.with_reanalyzed_policy_targets(fresh_policies)

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
        output_rows = torch.cat(
            (values[:, None], policy_logits),
            dim=1,
        ).float().cpu().tolist()
        roots = tuple(
            Evaluation(
                state=states[index],
                value_prefix=0.0,
                value=output_rows[index][0],
                policy_logits=output_rows[index][1:],
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
