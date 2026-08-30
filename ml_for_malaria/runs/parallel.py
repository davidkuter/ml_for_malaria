from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")


def replicate_worker_count(
    n_seeds: int,
    *,
    n_workers: int | None = None,
    serial: bool = False,
) -> int:
    """How many replicate fits to run at once.

    ``n_workers is None`` uses ``min(n_seeds, cpu_count)``. ``serial=True``
    (GPU trainers) or ``n_workers=1`` keeps the seed loop sequential.
    """
    if serial or n_workers == 1 or n_seeds <= 1:
        return 1
    cpus = os.cpu_count() or 1
    requested = cpus if n_workers is None else n_workers
    if requested < 1:
        raise ValueError(f"n_workers must be >= 1, got {requested}")
    return min(n_seeds, requested)


def map_replicates(
    fn: Callable[[int], T],
    seeds: Sequence[int],
    *,
    n_workers: int = 1,
) -> list[T]:
    """Call ``fn(seed)`` for each seed. ``n_workers>1`` uses a process pool.

    ``fn`` must be picklable (top-level in an importable module), not a lambda.
    """
    if n_workers <= 1:
        return [fn(seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, seeds))
