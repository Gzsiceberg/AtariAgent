"""Tensor batch representation sampled from trajectory replay."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from jaxtyping import Bool, Float, Int, UInt8
import torch
import torch.nn.functional as functional
from torch import Tensor

from .typecheck import runtime_typed


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A padded EfficientZero-style unroll batch.

    ``frames`` contains the initial stack context followed by one new frame
    per unroll action. ``action_mask`` identifies real
    action/reward steps. ``policy_mask`` identifies states with stored search
    policy targets. ``value_mask`` identifies states whose fixed-horizon return can
    be computed; a true terminal state has a valid zero-value target without
    an MCTS policy. Value-bootstrap fields carry compact real observations,
    stored bootstrap terms, and logical endpoint IDs so reanalysis can refresh
    TD endpoints without changing replay. The optional MCTS-bootstrap mask
    selects stale endpoints for search while recent endpoints retain direct values. Search values and
    zero-based transition ages (the number of newer replay transitions) are
    temporary metadata used to select
    EfficientZero V2's mixed value target before training. Reanalysis state
    IDs identify logical states across adjacent unrolls and overlapping blocks
    without relying on replay insertion order.
    """

    frames: UInt8[Tensor, "batch frames channels height width"]
    actions: Int[Tensor, "batch unroll 1"]
    rewards: Float[Tensor, "batch unroll"]
    policy_targets: Float[Tensor, "batch states actions"]
    value_targets: Float[Tensor, "batch states"]
    action_mask: Bool[Tensor, "batch unroll"]
    policy_mask: Bool[Tensor, "batch states"]
    value_mask: Bool[Tensor, "batch states"]
    indices: Int[Tensor, "batch"]
    importance_weights: Float[Tensor, "batch"]
    value_bootstrap_frames: (
        UInt8[Tensor, "batch bootstrap_frames channels height width"] | None
    ) = None
    value_bootstrap_values: Float[Tensor, "batch states"] | None = None
    value_bootstrap_discounts: Float[Tensor, "batch states"] | None = None
    value_bootstrap_mask: Bool[Tensor, "batch states"] | None = None
    value_bootstrap_state_ids: Int[Tensor, "batch states"] | None = None
    mcts_bootstrap_mask: Bool[Tensor, "batch states"] | None = None
    reanalysis_frames: (
        UInt8[Tensor, "batch reanalysis_frames channels height width"] | None
    ) = None
    search_value_targets: Float[Tensor, "batch states"] | None = None
    transition_ages: Int[Tensor, "batch"] | None = None
    reanalysis_state_ids: Int[Tensor, "batch states"] | None = None

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def unroll_steps(self) -> int:
        return self.actions.shape[1]

    @property
    def stack_size(self) -> int:
        return self.frames.shape[1] - self.unroll_steps

    def with_reanalysis_targets(
        self,
        *,
        value_targets: Float[Tensor, "batch states"],
        policy_targets: Float[Tensor, "batch states actions"],
        search_value_targets: Float[Tensor, "batch states"] | None = None,
    ) -> ReplayBatch:
        """Merge target-only asynchronous reanalysis output into this batch."""
        if value_targets.shape != self.value_targets.shape:
            raise ValueError("reanalyzed value targets have an invalid shape")
        if policy_targets.shape != self.policy_targets.shape:
            raise ValueError("reanalyzed policy targets have an invalid shape")
        if value_targets.device != self.value_targets.device:
            raise ValueError("reanalyzed value targets are on the wrong device")
        if policy_targets.device != self.policy_targets.device:
            raise ValueError("reanalyzed policy targets are on the wrong device")
        if search_value_targets is not None:
            if search_value_targets.shape != self.value_targets.shape:
                raise ValueError("search value targets have an invalid shape")
            if search_value_targets.device != self.value_targets.device:
                raise ValueError("search value targets are on the wrong device")
        return replace(
            self,
            value_targets=value_targets,
            policy_targets=policy_targets,
            search_value_targets=search_value_targets,
        )

    def with_selected_value_targets(
        self,
        *,
        mode: str,
        learner_step: int,
        collection_steps: int,
        mixed_start_step: int,
        freshness_threshold: int,
        preserve_mixed_value_freshness: bool = False,
    ) -> ReplayBatch:
        """Select EfficientZero V2 TD or search values for training."""
        if mode not in {"td", "search", "mixed"}:
            raise ValueError("value target mode must be td, search, or mixed")
        if not isinstance(preserve_mixed_value_freshness, bool):
            raise TypeError("preserve_mixed_value_freshness must be a boolean")
        for value, name in (
            (learner_step, "learner_step"),
            (collection_steps, "collection_steps"),
            (mixed_start_step, "mixed_start_step"),
            (freshness_threshold, "freshness_threshold"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if mode == "td" or (mode == "mixed" and learner_step < mixed_start_step):
            return self
        if self.search_value_targets is None:
            raise ValueError("batch has no MCTS search value targets")

        search_mask = self.policy_mask & self.value_mask
        if mode == "mixed":
            if self.transition_ages is None:
                raise ValueError("batch has no replay transition ages")
            if self.transition_ages.shape != (self.batch_size,):
                raise ValueError("transition ages have an invalid shape")
            effective_ages = self.effective_transition_ages(
                learner_step=learner_step,
                collection_steps=collection_steps,
                advance_during_final=not preserve_mixed_value_freshness,
            )
            sample_uses_search = effective_ages >= freshness_threshold
            search_mask = search_mask & sample_uses_search[:, None]
        selected = torch.where(
            search_mask,
            self.search_value_targets,
            self.value_targets,
        )
        return replace(self, value_targets=selected)

    def effective_transition_ages(
        self,
        *,
        learner_step: int,
        collection_steps: int,
        advance_during_final: bool = True,
    ) -> Tensor:
        """Optionally add learner-only updates to each replay-transition age."""
        if not isinstance(advance_during_final, bool):
            raise TypeError("advance_during_final must be a boolean")
        for value, name in (
            (learner_step, "learner_step"),
            (collection_steps, "collection_steps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.transition_ages is None:
            raise ValueError("batch has no replay transition ages")
        if self.transition_ages.shape != (self.batch_size,):
            raise ValueError("transition ages have an invalid shape")
        if torch.any(self.transition_ages < 0):
            raise ValueError("transition ages must be non-negative")
        if not advance_during_final:
            return self.transition_ages
        final_update_steps = max(learner_step - collection_steps, 0)
        return self.transition_ages + final_update_steps

    @staticmethod
    @runtime_typed
    def _policy_cross_entropy(
        logits: Float[Tensor, "batch actions"],
        target: Float[Tensor, "batch actions"],
    ) -> Float[Tensor, "batch"]:
        if logits.shape != target.shape:
            raise ValueError("policy logits and targets must have the same shape")
        return -(target * functional.log_softmax(logits, dim=-1)).sum(dim=-1)

    @staticmethod
    @runtime_typed
    def _scalar_loss(
        logits: Float[Tensor, "batch support"],
        target: Float[Tensor, "batch"],
        *,
        support_min: int,
        support_max: int,
        epsilon: float = 0.001,
    ) -> Float[Tensor, "batch"]:
        if support_min >= support_max:
            raise ValueError("support_min must be less than support_max")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        expected_size = support_max - support_min + 1
        if logits.ndim != target.ndim + 1 or logits.shape[:-1] != target.shape:
            raise ValueError("logits must have target.shape followed by a support axis")
        if logits.shape[-1] != expected_size:
            raise ValueError(
                f"expected {expected_size} support logits, got {logits.shape[-1]}"
            )

        transformed = (
            target.sign() * (torch.sqrt(target.abs() + 1.0) - 1.0) + epsilon * target
        )
        transformed = transformed.clamp(support_min, support_max) - support_min
        lower = transformed.floor().long()
        upper = transformed.ceil().long()
        upper_weight = transformed - lower
        lower_weight = 1.0 - upper_weight

        # Cross entropy against an index is -log_softmax(logits)[index].
        # Compute the support-wide reduction once, then gather both adjacent
        # support atoms. Low-precision logits are reduced in FP32 for safety.
        reduction_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype
        )
        log_probabilities = functional.log_softmax(
            logits, dim=-1, dtype=reduction_dtype
        )
        lower_loss = -log_probabilities.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
        upper_loss = -log_probabilities.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
        return lower_weight * lower_loss + upper_weight * upper_loss

    def without_value_bootstraps(self) -> ReplayBatch:
        """Drop direct-value reanalysis metadata."""
        return replace(
            self,
            value_bootstrap_frames=None,
            value_bootstrap_values=None,
            value_bootstrap_discounts=None,
            value_bootstrap_mask=None,
            value_bootstrap_state_ids=None,
            mcts_bootstrap_mask=None,
        )

    def without_reanalysis_metadata(self) -> ReplayBatch:
        """Drop all temporary target metadata before learner transfer."""
        return replace(
            self.without_value_bootstraps(),
            reanalysis_frames=None,
            search_value_targets=None,
            transition_ages=None,
            reanalysis_state_ids=None,
        )

    def pin_memory(self) -> ReplayBatch:
        """Copy CPU tensors into page-locked memory for asynchronous transfer."""
        shared = self.reanalysis_frames
        pinned_shared = None if shared is None else shared.pin_memory()
        bootstrap_offset = (
            0
            if shared is None or self.value_bootstrap_frames is None
            else shared.shape[1] - self.value_bootstrap_frames.shape[1]
        )
        values: dict[str, Tensor | None] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if pinned_shared is not None and field.name == "reanalysis_frames":
                values[field.name] = pinned_shared
            elif pinned_shared is not None and field.name == "frames":
                values[field.name] = pinned_shared[:, : self.frames.shape[1]]
            elif (
                pinned_shared is not None
                and field.name == "value_bootstrap_frames"
                and self.value_bootstrap_frames is not None
            ):
                values[field.name] = pinned_shared[
                    :,
                    bootstrap_offset : bootstrap_offset
                    + self.value_bootstrap_frames.shape[1],
                ]
            else:
                values[field.name] = (
                    value.pin_memory() if isinstance(value, Tensor) else value
                )
        return ReplayBatch(**values)

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Keep CUDA tensor storage alive on a consuming stream."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor) and value.device.type == "cuda":
                value.record_stream(stream)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
        keep_indices_on_cpu: bool = False,
    ) -> ReplayBatch:
        """Move batch tensors to ``device``, optionally with async copies."""
        return ReplayBatch(
            **{
                field.name: (
                    value
                    if not isinstance(value, Tensor)
                    or (field.name == "indices" and keep_indices_on_cpu)
                    else value.to(device, non_blocking=non_blocking)
                )
                for field in fields(self)
                for value in (getattr(self, field.name),)
            }
        )


__all__ = ["ReplayBatch"]
