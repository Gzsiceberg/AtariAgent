import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from atariagent.typecheck import (
    runtime_typed,
    runtime_typechecking_enabled,
    set_runtime_typechecking,
)


@runtime_typed
def floating_identity(value: Float[Tensor, "batch"]) -> Tensor:
    return value


def test_runtime_tensor_checks_can_be_disabled_for_production() -> None:
    original = runtime_typechecking_enabled()
    try:
        set_runtime_typechecking(True)
        with pytest.raises(Exception):
            floating_identity(torch.ones(2, dtype=torch.long))

        set_runtime_typechecking(False)
        value = torch.ones(2, dtype=torch.long)
        assert floating_identity(value) is value
    finally:
        set_runtime_typechecking(original)
