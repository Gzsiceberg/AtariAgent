# Agent Instructions

- This repository uses [`uv`](https://docs.astral.sh/uv/) to manage Python dependencies and the virtual environment.
- Add or remove dependencies with `uv add` and `uv remove`; do not use `pip install` directly.
- Run Python commands and project tools with `uv run`.
- Keep `pyproject.toml` and `uv.lock` synchronized, and commit dependency changes to both files.
- Use [Hydra](https://hydra.cc/) for configuration files and configuration management.
- Use [`pytest`](https://docs.pytest.org/) for writing and running tests.
- Prefer [`einops`](https://einops.rocks/) for most tensor shape operations instead of manual PyTorch shape manipulation.
- Do not update the user's plan unless explicitly asked to do so.
- Do not commit changes unless the user explicitly asks you to commit them.

## Reference Repositories

- `~/repos/EfficientZeroV2`
- `~/repos/EfficientZero`
