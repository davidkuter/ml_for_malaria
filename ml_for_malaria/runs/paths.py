from __future__ import annotations

from pathlib import Path

from ml_for_malaria.schemas import Architecture

_DIR_SLUGS = {
    Architecture.XGBOOST: "xgb",
    Architecture.RANDOM_FOREST: "rf",
}

DEFAULT_N_REP = 10
DEFAULT_SEED_START = 42


def architecture_dir_slug(architecture: str) -> str:
    """Short folder token for an architecture (``xgboost`` → ``xgb``)."""
    return _DIR_SLUGS.get(architecture, architecture)


def run_dirname(
    architecture: str,
    split: str,
    *,
    charge_method: str | None = None,
    hpo: bool = False,
    yscramble: bool = False,
) -> str:
    """Experiment folder: ``{arch}_{split}`` plus optional charge, ``hpo``, ``yscramble``."""
    parts = [architecture_dir_slug(architecture), split]
    if charge_method:
        parts.append(charge_method)
    if hpo:
        parts.append("hpo")
    if yscramble:
        parts.append("yscramble")
    return "_".join(parts)


def seed_dir_name(seed: int) -> str:
    """Replicate subdirectory under an experiment folder (``seed_42``)."""
    return f"seed_{seed}"


def replicate_seeds(n_rep: int, *, start: int = DEFAULT_SEED_START) -> tuple[int, ...]:
    """Consecutive seeds for ``n_rep`` independent train/test replicates."""
    if n_rep < 1:
        raise ValueError(f"n_rep must be >= 1, got {n_rep}")
    return tuple(range(start, start + n_rep))


def resolve_run_dir(
    parent: str | Path,
    architecture: str,
    split: str,
    *,
    charge_method: str | None = None,
    seed: int | None = None,
    hpo: bool = False,
    yscramble: bool = False,
) -> Path:
    """Resolve the run directory under ``parent``.

    Without ``seed`` this is the experiment folder (legacy ``xgb_random``).
    With ``seed`` artifacts live in ``{experiment}/seed_{seed}/``.
    ``hpo=True`` uses ``{arch}_{split}_hpo`` so tuned runs do not overwrite
    the fixed-recipe comparison. ``yscramble=True`` appends ``_yscramble``
    (train labels permuted; test labels unchanged).
    """
    experiment = Path(parent) / run_dirname(
        architecture,
        split,
        charge_method=charge_method,
        hpo=hpo,
        yscramble=yscramble,
    )
    if seed is None:
        return experiment
    return experiment / seed_dir_name(seed)


def completed_run_dirs(parent: str | Path) -> list[Path]:
    """Run directories under ``parent`` that contain a training ``report.json``.

    Finds both a report on an immediate child and reports one level down
    (``{experiment}/seed_{n}/``).
    """
    from ml_for_malaria.runs.checkpoints import RunCheckpointer

    root = Path(parent)
    if not root.exists():
        return []
    name = RunCheckpointer.REPORT_JSON
    found: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / name).exists():
            found.append(path)
            continue
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / name).exists():
                found.append(child)
    return found
