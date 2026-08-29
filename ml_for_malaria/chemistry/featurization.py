import datamol as dm
import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdchem import MolSanitizeException
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganFeatureAtomInvGen,
    GetMorganGenerator,
    GetRDKitFPGenerator,
)

from ml_for_malaria.schemas import CleanedTrainingData, FingerprintFeatures

DEFAULT_FP_SIZE = 2048
_MOL_PARSE_ERRORS = (TypeError, ValueError, RuntimeError, MolSanitizeException)


def sanitize_smiles(smiles: str, as_mol: bool = False) -> str | Chem.Mol | None:
    """Sanitise a SMILES string and optionally return an RDKit molecule.

    Parses, strips salts/solvents, keeps the largest remaining fragment, then
    standardises and uncharges. Returns None if the input is not a string or
    sanitization fails.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        logger.warning(f'"{smiles}" failed sanitization')
        return None
    try:
        mol = dm.to_mol(smiles)
        if mol is None:
            raise ValueError("unparseable SMILES")
        mol = dm.remove_salts_solvents(mol, dont_remove_everything=True)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("salt stripping removed the molecule")
        if len(Chem.GetMolFrags(mol)) > 1:
            mol = dm.keep_largest_fragment(mol)
        mol = dm.standardize_mol(mol, uncharge=True)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("standardization failed")
        if as_mol:
            return mol
        return Chem.MolToSmiles(mol)
    except _MOL_PARSE_ERRORS:
        logger.warning(f'"{smiles}" failed sanitization')
        return None


def featurize_smiles(
    smiles: list[str] | pd.Series, fp_generator, sanitize: bool = False
) -> pd.DataFrame:
    """Featurize SMILES with an RDKit fingerprint generator.

    Rows that fail to parse are omitted. The index contains only the SMILES that
    succeeded, so callers must not assume ``len(result) == len(smiles)``.
    """
    smiles = list(smiles)
    logger.info(f"Featurizing {len(smiles)} SMILES")

    rows: list[np.ndarray] = []
    index: list[str] = []
    for smi in smiles:
        mol = None
        if sanitize:
            mol = sanitize_smiles(smi, as_mol=True)
        else:
            try:
                mol = Chem.MolFromSmiles(smi) if smi else None
            except _MOL_PARSE_ERRORS:
                mol = None
        if mol is None:
            logger.warning(f'"{smi}" failed featurization')
            continue
        fps = fp_generator.GetFingerprint(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fps, array)
        rows.append(array)
        index.append(smi)

    if not rows:
        return FingerprintFeatures.validate(pd.DataFrame(index=index))

    return FingerprintFeatures.validate(
        pd.DataFrame(rows, columns=list(range(len(rows[0]))), index=index)
    )


def get_fingerprint_generators(fp_size: int = DEFAULT_FP_SIZE) -> dict:
    """Named fingerprint generators used for training and inference."""
    return {
        "Morgan2Bits": GetMorganGenerator(radius=2, fpSize=fp_size),
        "Morgan2FeatBits": GetMorganGenerator(
            radius=2,
            fpSize=fp_size,
            atomInvariantsGenerator=GetMorganFeatureAtomInvGen(),
        ),
        "Morgan3Bits": GetMorganGenerator(radius=3, fpSize=fp_size),
        "RDKit": GetRDKitFPGenerator(fpSize=fp_size),
        "AtomPair": GetAtomPairGenerator(fpSize=fp_size),
    }


def get_fingerprint_generator(name: str, fp_size: int = DEFAULT_FP_SIZE):
    """Look up a fingerprint generator by the name stored in model metadata."""
    generators = get_fingerprint_generators(fp_size=fp_size)
    if name not in generators:
        supported = ", ".join(sorted(generators))
        raise ValueError(f"Unknown fingerprint {name!r}. Supported: {supported}")
    return generators[name]


def _process_atom_pair_bits(
    info: dict[int, tuple[tuple[int, int]]],
) -> dict[int, set[int]]:
    """Collapse atom pairs into unique atoms encoded in AtomPair fingerprints."""
    new_map = {}
    for bit, atom_pairs in info.items():
        unique_atoms = set()
        for atom1, atom2 in atom_pairs:
            unique_atoms.add(atom1)
            unique_atoms.add(atom2)
        new_map[bit] = unique_atoms
    return new_map


def get_bit_atom_map(mol: Chem.Mol, fp_generator) -> dict[int, set[int]] | None:
    """Map fingerprint bits to atom indices. AtomPair is implemented; others return None."""
    ao = AllChem.AdditionalOutput()
    ao.CollectBitInfoMap()
    _ = fp_generator.GetFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()

    fp_gen_type = fp_generator.GetOptions().__str__()
    if "AtomPair" in fp_gen_type:
        return _process_atom_pair_bits(info=info)
    return None


def encode_binary_labels(
    series: pd.Series,
    active_label: str = "Active",
    inactive_label: str = "Inactive",
) -> pd.Series:
    """Map two string labels to 1/0. Raises if unexpected values are present."""
    known = series.dropna()
    unexpected = known.loc[~known.isin([active_label, inactive_label])]
    if not unexpected.empty:
        extra = sorted(unexpected.unique().tolist())
        raise ValueError(
            f"Unexpected labels: {extra}. "
            f"Expected {active_label!r} or {inactive_label!r}."
        )
    return series.map({active_label: 1, inactive_label: 0})


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize SMILES, drop failures, and drop SMILES with conflicting labels."""
    smiles = CleanedTrainingData.SMILES
    label = CleanedTrainingData.LABEL
    input_smiles = CleanedTrainingData.INPUT_SMILES
    if smiles not in df.columns or label not in df.columns:
        raise ValueError(
            f"Training dataframe must contain {smiles} and {label} columns"
        )

    cleaned = df.copy()
    cleaned[input_smiles] = cleaned[smiles]
    cleaned[smiles] = cleaned[input_smiles].map(
        lambda smi: sanitize_smiles(smi, as_mol=False)
    )
    n_failed = int(cleaned[smiles].isna().sum())
    if n_failed:
        logger.warning(f"Dropping {n_failed} rows that failed SMILES sanitization")
    cleaned = cleaned.dropna(subset=[smiles, label])
    cleaned[label] = cleaned[label].astype(int)

    conflict_mask = cleaned.groupby(smiles)[label].transform("nunique").gt(1)
    n_conflicts = int(cleaned.loc[conflict_mask, smiles].nunique())
    if n_conflicts:
        logger.warning(
            f"Dropping {n_conflicts} SMILES with conflicting labels after sanitization"
        )
        cleaned = cleaned.loc[~conflict_mask]

    n_dupes = int(cleaned.duplicated(subset=[smiles]).sum())
    if n_dupes:
        logger.info(f"Dropping {n_dupes} duplicate sanitized SMILES (labels agreed)")
    return CleanedTrainingData.validate(
        cleaned.drop_duplicates(subset=[smiles], keep="first").reset_index(drop=True)
    )
