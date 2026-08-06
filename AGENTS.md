# Agent Instructions

- This repository uses [`uv`](https://docs.astral.sh/uv/) to manage Python dependencies and the virtual environment.
- Add or remove dependencies with `uv add` and `uv remove`; do not use `pip install` directly.
- Run Python commands and project tools with `uv run`.
- Keep `pyproject.toml` and `uv.lock` synchronized, and commit dependency changes to both files.
- Use [Hydra](https://hydra.cc/) for configuration files and configuration management.
- Use [`pytest`](https://docs.pytest.org/) for writing and running tests.
- Prefer [`einops`](https://einops.rocks/) for most tensor shape operations instead of manual PyTorch shape manipulation.
- Do not use LaTeX; use plain-text notation because LaTeX cannot render in the terminal.

## Reference Repositories

- `~/repos/EfficientZeroV2`
- `~/repos/EfficientZero`
