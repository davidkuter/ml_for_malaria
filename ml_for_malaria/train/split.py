from typing import Protocol

import datamol as dm
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


class Splitter(Protocol):
    """Train/test split strategy. Returns integer positional indices."""

    name: str

    def split(
        self,
        smiles: pd.Series,
        labels: pd.Series,
        test_size: float,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        """Return (train_indices, test_indices) aligned with ``smiles`` / ``labels``."""
        ...


class RandomSplitter:
    """Stratified random split. Every fingerprint type shares these compound indices."""

    name = "random"

    def split(
        self,
        smiles: pd.Series,
        labels: pd.Series,
        test_size: float,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        del smiles  # same compounds for every fingerprint; grouping comes later
        positions = pd.RangeIndex(len(labels))
        train_idx, test_idx = train_test_split(
            positions,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        return train_idx.tolist(), test_idx.tolist()


def murcko_group_keys(smiles: pd.Series) -> pd.Series:
    """Bemis–Murcko scaffold keys; empty scaffolds are unique per row."""
    smiles = smiles.reset_index(drop=True)
    mols = smiles.map(dm.to_mol)
    scaffolds = mols.map(
        lambda mol: dm.to_scaffold_murcko(mol) if mol is not None else None
    )
    keys = scaffolds.map(
        lambda scf: dm.to_smiles(scf)
        if scf is not None and scf.GetNumAtoms() > 0
        else None
    )
    empty = keys.isna() | (keys.astype(str).str.len() == 0)
    singletons = "no_scaffold:" + smiles.index.astype(str)
    return keys.where(~empty, singletons)


class ScaffoldSplitter:
    """Split by Bemis–Murcko scaffold groups using sklearn GroupShuffleSplit.

    Whole scaffold groups stay on one side of the split (seeded, approximate
    ``test_size``). Labels are not stratified: sklearn cannot keep groups
    intact and match class balance at the same time.
    """

    name = "scaffold"

    def split(
        self,
        smiles: pd.Series,
        labels: pd.Series,
        test_size: float,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        smiles = smiles.reset_index(drop=True)
        labels = labels.reset_index(drop=True)
        if len(smiles) != len(labels):
            raise ValueError("smiles and labels must have the same length")
        positions = pd.RangeIndex(len(smiles))
        groups = murcko_group_keys(smiles)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_idx, test_idx = next(
            splitter.split(positions, labels, groups=groups)
        )
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                "Scaffold split produced an empty train or test set. "
                "A single large Bemis–Murcko group may exceed test_size; "
                "use split='random' or a different test_size."
            )
        return train_idx.tolist(), test_idx.tolist()


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
