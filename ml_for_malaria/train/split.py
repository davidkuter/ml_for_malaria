from typing import Protocol

import numpy as np
from sklearn.model_selection import train_test_split


class Splitter(Protocol):
    """Train/test split strategy. Returns integer positional indices."""

    name: str

    def split(
        self,
        smiles: list[str],
        labels: list[int],
        test_size: float,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        """Return (train_indices, test_indices) into the aligned smiles/labels lists."""
        ...


class RandomSplitter:
    """Stratified random split. Every fingerprint type shares these compound indices."""

    name = "random"

    def split(
        self,
        smiles: list[str],
        labels: list[int],
        test_size: float,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        indices = np.arange(len(smiles))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        return train_idx.tolist(), test_idx.tolist()


class ScaffoldSplitter:
    """Split by Bemis–Murcko scaffold groups, then assign groups to train/test.

    Intended contract (not yet implemented):
    1. Compute a scaffold key for each SMILES (e.g. Bemis–Murcko).
    2. Assign whole scaffold groups to train or test so no scaffold appears in both.
    3. Honour ``test_size`` approximately at the compound (or group) level.
    4. Prefer preserving class balance when assigning groups.
    """

    name = "scaffold"

    def split(
        self,
        smiles: list[str],
        labels: list[int],
        test_size: float,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        raise NotImplementedError(
            "Scaffold splitting is not implemented yet. Use split='random', "
            "or implement ScaffoldSplitter.split to group by Bemis–Murcko "
            "scaffold and assign groups to train/test."
        )


_SPLITTERS: dict[str, type] = {
    "random": RandomSplitter,
    "scaffold": ScaffoldSplitter,
}


def get_splitter(name: str) -> Splitter:
    """Return a splitter instance by name."""
    if name not in _SPLITTERS:
        supported = ", ".join(sorted(_SPLITTERS))
        raise ValueError(f"Unknown split {name!r}. Supported: {supported}")
    return _SPLITTERS[name]()
