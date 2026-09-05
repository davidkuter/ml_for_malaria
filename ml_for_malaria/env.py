"""Load gitignored repo-root ``.local.env`` into ``os.environ`` (process wins)."""

from __future__ import annotations

import os
from pathlib import Path

_LOCAL_ENV_NAME = ".local.env"
_LOADED = False


def repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this file) until ``pyproject.toml`` is found."""
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def local_env_path(start: Path | None = None) -> Path:
    return repo_root(start) / _LOCAL_ENV_NAME


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; ignore blanks and ``#`` comments."""
    values: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_local_env(*, force: bool = False, start: Path | None = None) -> Path | None:
    """Load ``.local.env`` into ``os.environ`` for keys that are not already set.

    Returns the path loaded, or ``None`` if the file is missing.
    Process / shell environment always wins over the file.
    """
    global _LOADED
    if _LOADED and not force:
        path = local_env_path(start)
        return path if path.exists() else None
    path = local_env_path(start)
    if not path.exists():
        _LOADED = True
        return None
    for key, value in parse_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = value
    _LOADED = True
    return path
