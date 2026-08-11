"""Runtime tensor type-checking controls.

The project keeps jaxtyping annotations in production, but expensive beartype
checks can be disabled for hot training paths.  Tests and interactive use keep
checks enabled by default.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from beartype import beartype
from jaxtyping import jaxtyped


P = ParamSpec("P")
R = TypeVar("R")
_RUNTIME_TYPECHECKING_ENABLED = True


def runtime_typechecking_enabled() -> bool:
    """Return whether decorated calls currently run jaxtyping/beartype checks."""
    return _RUNTIME_TYPECHECKING_ENABLED


def set_runtime_typechecking(enabled: bool) -> None:
    """Enable or disable runtime checks without removing static annotations."""
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a boolean")
    global _RUNTIME_TYPECHECKING_ENABLED
    _RUNTIME_TYPECHECKING_ENABLED = enabled


def runtime_typed(function: Callable[P, R]) -> Callable[P, R]:
    """Decorate ``function`` with checks that can be bypassed at runtime."""
    checked = jaxtyped(typechecker=beartype)(function)

    @wraps(function)
    def dispatch(*args: P.args, **kwargs: P.kwargs) -> R:
        if _RUNTIME_TYPECHECKING_ENABLED:
            return checked(*args, **kwargs)
        return function(*args, **kwargs)

    return dispatch


__all__ = [
    "runtime_typed",
    "runtime_typechecking_enabled",
    "set_runtime_typechecking",
]
