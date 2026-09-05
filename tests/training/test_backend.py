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
@pytest.mark.parametrize("autotune", [False, True])
def test_backend_autotuning_is_independent_of_determinism_and_tf32(
    train_script, backend, deterministic, autotune
):
    _, cudnn, matmul, precision = backend
    train_script.configure_training_backend(deterministic, cudnn_benchmark=autotune)
    assert cudnn.benchmark is autotune
    assert cudnn.deterministic is deterministic
    assert cudnn.allow_tf32 is (not deterministic)
    assert matmul.allow_tf32 is (not deterministic)
    assert precision == ["highest" if deterministic else "high"]


def test_default_disables_autotuning_without_disabling_tf32(train_script, backend):
    _, cudnn, matmul, precision = backend
    train_script.configure_training_backend(False)
    assert cudnn.benchmark is False
    assert cudnn.allow_tf32 is True
    assert matmul.allow_tf32 is True
    assert precision == ["high"]


def test_backend_is_unchanged_without_cuda(train_script, backend):
    fake_torch, cudnn, matmul, precision = backend
    fake_torch.cuda.is_available = lambda: False
    train_script.configure_training_backend(False, cudnn_benchmark=True)
    assert vars(cudnn) == {"benchmark": None, "deterministic": None, "allow_tf32": None}
    assert matmul.allow_tf32 is None
    assert precision == []
