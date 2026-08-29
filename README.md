# ml_for_malaria

Repository storing my dabbles in using ML for Malaria-related research.

## Setup

This project is packaged with [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

That creates `.venv`, installs runtime and `dev` dependencies, and installs the package in editable mode. `activate.sh` is a local helper (gitignored) for WSL on a Windows drive; source it instead of `uv sync` when you have it.

```bash
uv run pytest
uv run ruff check .
```

Add a runtime dependency with `uv add <package>`, or a dev-only tool with `uv add --dev <package>`. `uv.lock` is the pinned resolution; regenerate it with `uv lock`.
