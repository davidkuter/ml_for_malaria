# ml_for_malaria

Repository storing my dabbles in using ML for Malaria-related research.

## Setup

This project is packaged with [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

That creates `.venv`, installs runtime and `dev` dependencies, and installs the package in editable mode. `activate.sh` is a local helper (gitignored) for WSL on a Windows drive; source it instead of `uv sync` when you have it. ChemBERTa and Chemprop need the optional `dl` extra (torch, transformers, chemprop):

```bash
uv sync --extra dl
```

Chemprop `charge_method="nagl"` needs the OpenFF extra (GitHub sources; official PyPI wheels are yanked):

```bash
uv sync --extra dl --extra nagl
```

On Windows, enable Git long paths first (`git config --global core.longpaths true`) or the NAGL clone fails on long test filenames. Without the extra, use `charge_method="gasteiger"` or `None`.

```bash
uv run pytest
uv run ruff check .
```

Add a runtime dependency with `uv add <package>`, or a dev-only tool with `uv add --dev <package>`. `uv.lock` is the pinned resolution; regenerate it with `uv lock`.

## Architecture

Package layout, import direction, and where new code belongs: [docs/architecture.md](docs/architecture.md). `ml_for_malaria.train` is trainers only; chemistry, splits, run I/O, and reports live beside it.
