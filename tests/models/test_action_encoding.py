"""V1 action-plane ablation and saved architecture selection."""

from pathlib import Path

from einops import repeat
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from atariagent.agent import AtariAgent
from atariagent.evaluation import load_agent_checkpoint
from atariagent.models import DynamicsNetwork
from atariagent.training.config import register_train_agent_config


@pytest.mark.parametrize("action_embedding", [True, False])
def test_hydra_action_embedding_override(action_embedding):
    register_train_agent_config()
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="train_agent", overrides=[
            f"model.action_embedding={str(action_embedding).lower()}",
        ])
    assert config.model.action_embedding is action_embedding
    assert OmegaConf.to_object(config).model.action_embedding is action_embedding


def test_raw_action_plane_matches_v1_and_receives_gradients():
    torch.manual_seed(5)
    model = DynamicsNetwork(6, action_embedding=False)
    assert model.transition[0].in_channels == 65
    assert not any(key.startswith("action_") for key in model.state_dict())
    state = torch.randn(3, 64, 6, 6, requires_grad=True)
    actions = torch.tensor([[0], [1], [5]])
    inputs = []
    hook = model.transition[0].register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())
    )
    try:
        next_state, _, _ = model(state, actions)
    finally:
        hook.remove()
    expected_plane = repeat(actions.float() / 6, "b c -> b c h w", h=6, w=6)
    torch.testing.assert_close(inputs[0][:, :64], state.detach())
    torch.testing.assert_close(inputs[0][:, 64:], expected_plane)
    next_state.square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    action_weight_gradient = model.transition[0].weight.grad[:, -1]
    assert torch.isfinite(action_weight_gradient).all()
    assert action_weight_gradient.abs().sum() > 0


@pytest.mark.parametrize("action_embedding", [True, False])
def test_action_encoding_checkpoint_round_trip(tmp_path, action_embedding):
    register_train_agent_config()
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="train_agent", overrides=[
            "environment.id=ALE/Qbert-v5", "training.precision=fp32",
            f"model.action_embedding={str(action_embedding).lower()}",
        ])
    original = AtariAgent(12, 6, action_embedding=action_embedding)
    path = tmp_path / "checkpoint.pt"
    checkpoint = {"config": OmegaConf.to_container(config, resolve=True)}
    for name in ("representation", "dynamics", "prediction"):
        checkpoint[name] = getattr(original, f"{name}_network").state_dict()
    torch.save(checkpoint, path)
    restored, saved_config = load_agent_checkpoint(
        path, action_space_size=6, device=torch.device("cpu")
    )
    assert restored.dynamics_network.action_embedding is action_embedding
    assert saved_config["model"]["action_embedding"] is action_embedding
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_action_embedding_requires_boolean():
    with pytest.raises(TypeError, match="action_embedding"):
        DynamicsNetwork(6, action_embedding="false")
