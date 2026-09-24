"""Backend policy tests without requiring CUDA or changing global torch flags."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def train_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_agent.py"
    spec = importlib.util.spec_from_file_location("train_backend_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(train_script, monkeypatch):
    cudnn = SimpleNamespace(benchmark=None, deterministic=None, allow_tf32=None)
    matmul = SimpleNamespace(allow_tf32=None)
    precision = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(matmul=matmul)),
        set_float32_matmul_precision=precision.append,
    )
    monkeypatch.setattr(train_script, "torch", fake_torch)
    return fake_torch, cudnn, matmul, precision


@pytest.mark.parametrize("deterministic", [False, True])
def test_backend_only_changes_cudnn_determinism(
    train_script, backend, deterministic
):
    _, cudnn, matmul, precision = backend
    train_script.configure_training_backend(deterministic)
    assert cudnn.benchmark is None
    assert cudnn.deterministic is deterministic
    assert cudnn.allow_tf32 is None
    assert matmul.allow_tf32 is None
    assert precision == []


def test_backend_is_unchanged_without_cuda(train_script, backend):
    fake_torch, cudnn, matmul, precision = backend
    fake_torch.cuda.is_available = lambda: False
    train_script.configure_training_backend(True)
    assert vars(cudnn) == {"benchmark": None, "deterministic": None, "allow_tf32": None}
    assert matmul.allow_tf32 is None
    assert precision == []
