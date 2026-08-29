from __future__ import annotations

from pathlib import Path

from ml_for_malaria.schemas import Architecture

_XGBOOST_DIR = "xgb"


def architecture_dir_slug(architecture: str) -> str:
    """Short folder token for an architecture (``xgboost`` → ``xgb``)."""
    if architecture == Architecture.XGBOOST:
        return _XGBOOST_DIR
    return architecture


def run_dirname(
    architecture: str,
    split: str,
    *,
    charge_method: str | None = None,
) -> str:
    """Build ``{arch}_{split}`` or ``{arch}_{split}_{charge}``."""
    parts = [architecture_dir_slug(architecture), split]
    if charge_method:
        parts.append(charge_method)
    return "_".join(parts)


def resolve_run_dir(
    parent: str | Path,
    architecture: str,
    split: str,
    *,
    charge_method: str | None = None,
) -> Path:
    """Resolve the run subdirectory under ``parent`` (e.g. ``runs/``)."""
    return Path(parent) / run_dirname(architecture, split, charge_method=charge_method)


def completed_run_dirs(parent: str | Path) -> list[Path]:
    """Subdirectories of ``parent`` that contain a training ``report.json``."""
    from ml_for_malaria.runs.checkpoints import RunCheckpointer

    root = Path(parent)
    if not root.exists():
        return []
    name = RunCheckpointer.REPORT_JSON
    return sorted(
        path for path in root.iterdir() if path.is_dir() and (path / name).exists()
    )
