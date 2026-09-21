"""Bounded snapshot history for delayed replay-reanalysis publications."""

from collections.abc import Callable, Mapping

import torch

TargetState = dict[str, torch.Tensor]


class TargetDelay:
    """Capture only snapshots needed by future fixed-interval publications.

    A publication at update t uses weights from max(0, t - delay).
    Snapshots supplied by capture must be independent copies of learner weights.
    """

    def __init__(self, interval: int, delay: int, initial: Mapping[str, torch.Tensor]):
        for name, value in (("interval", interval), ("delay", delay)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"target {name} must be an integer")
        if interval <= 0 or delay < 0:
            raise ValueError("target interval must be positive and delay non-negative")
        self.interval = interval
        self.delay = delay
        self.pending: dict[int, TargetState] = {0: dict(initial)}

    def advance(self, update: int, capture: Callable[[], TargetState]) -> TargetState | None:
        """Call after every learner update; return weights only when due."""
        if (update + self.delay) % self.interval == 0:
            self.pending[update] = capture()
        if update % self.interval:
            return None
        source = max(0, update - self.delay)
        state = self.pending[source]
        self.pending = {step: weights for step, weights in self.pending.items() if step >= source}
        return state

    def state_dict(self) -> dict[str, object]:
        return {"interval": self.interval, "delay": self.delay, "pending": dict(self.pending)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("interval") != self.interval or state.get("delay") != self.delay:
            raise ValueError("snapshot target update interval/delay do not match configuration")
        pending = state.get("pending")
        if not isinstance(pending, dict) or not pending:
            raise ValueError("snapshot target delay history is invalid")
        for step, weights in pending.items():
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("snapshot target delay step is invalid")
            if not isinstance(weights, dict) or not all(
                isinstance(name, str) and isinstance(value, torch.Tensor)
                for name, value in weights.items()
            ):
                raise ValueError("snapshot target delay weights are invalid")
        self.pending = dict(pending)
