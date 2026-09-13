"""Acting precision must match native target reanalysis, including decoding."""

import pytest
import torch

from atariagent.agent import AtariAgent, categorical_to_scalar
from atariagent.search import SearchConfig


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_scalar_decoding_uses_fp32(dtype):
    logits = torch.full((2, 601), -100.0, dtype=dtype)
    logits[0, 301] = 100.0
    logits[1, 305] = 100.0
    actual = categorical_to_scalar(logits)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, categorical_to_scalar(logits.float()))
    assert actual[0] > 0
    assert actual[1] > actual[0]


@pytest.mark.parametrize("precision,dtype", [("fp32", torch.float32), ("bf16", torch.bfloat16)])
def test_act_precision_covers_root_and_recurrent_search(precision, dtype):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        agent = AtariAgent(12, 3, precision=precision, search_config=SearchConfig(num_simulations=2))
        seen = {"representation": [], "prediction": [], "dynamics": []}
        handles = [
            agent.representation_network.register_forward_hook(
                lambda _m, _a, out: seen["representation"].append(out.dtype)
            ),
            agent.prediction_network.register_forward_hook(
                lambda _m, _a, out: seen["prediction"].append(out[1].dtype)
            ),
            agent.dynamics_network.register_forward_hook(
                lambda _m, _a, out: seen["dynamics"].append(out[2].dtype)
            ),
        ]
        try:
            # The agent's explicit precision must override an outer context.
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = agent.act(torch.zeros(2, 12, 96, 96))
            assert seen["representation"] == [dtype]
            assert seen["prediction"] == [dtype] * 3
            assert seen["dynamics"] == [dtype] * 2
            assert len(output.actions) == 2
            assert all(torch.isfinite(torch.tensor(output.predicted_values)))
            assert agent.training
            assert agent.representation_network.training
            assert all(p.dtype == torch.float32 for p in agent.parameters())
            assert not torch.is_autocast_enabled("cpu")
        finally:
            for handle in handles:
                handle.remove()
    finally:
        torch.set_num_threads(original_threads)


def test_agent_rejects_unsupported_precision():
    with pytest.raises(ValueError, match="precision"):
        AtariAgent(12, 3, precision="fp16")
