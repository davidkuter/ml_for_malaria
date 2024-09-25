import datamol as dm
import numpy as np
import pandas as pd

from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator


def sanitize_smiles(smiles: str, as_mol=False) -> str | Chem.Mol | None:
    """
    Sanitise a SMILES string and (optionally) return a RDKit molecule object

    :param smiles: SMILES string to sanitise
    :param as_mol: Boolean flag, if True return RDKit molecule object instead of SMILES string
    :return: Sanitised SMILES string or RDKit molecule object or None if sanitization fails
    """
    if isinstance(smiles, str):
        mol = dm.to_mol(smiles)
        mol = dm.standardize_mol(mol)
        if as_mol is True:
            return mol
        else:
            return Chem.MolToSmiles(mol)
    else:
        logger.warning(f'"{smiles}" failed sanitization')
        return None


def featurize_smiles(
        smiles: list[str], fp_generator: rdFingerprintGenerator, sanitize: bool = False
) -> pd.DataFrame:
    """
    Featurize a list of SMILES strings using a RDKit fingerprint generator. Optionally sanitise the SMILES strings

    :param smiles: SMILES strings to featurize
    :param fp_generator: RDKit fingerprint generator object
    :param sanitize: Boolean flag, if True sanitise the SMILES strings
    :return: A dataframe containing the features
    """
    logger.info(f"Featurizing {len(smiles)} SMILES")

    # Generate features
    features = []
    for smi in smiles:
        if sanitize:
            mol = sanitize_smiles(smi, as_mol=True)
        else:
            mol = Chem.MolFromSmiles(smi)
        if mol:
            fps = fp_generator.GetFingerprint(mol)
            array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fps, array)
            features.append(array)

    return pd.DataFrame(
        features, columns=[i for i in range(len(features[0]))], index=smiles
    )


def _process_atom_pair_bits(
        info: dict[int, tuple[tuple[int, int]]],
) -> dict[int, set[int]]:
    """
    Collapses atom pairs into unique atoms encoded in AtomPair fingerprints

    :param info: Dictionary with bit features as keys and tuples of atom pairs as values
                 e.g. {0: ((1, 2), (2, 3)), 1: ((3, 4), (5, 6))}
    :return: Dictionary with bit features as keys and sets of atom indices as values associated with the bit
                e.g. {0: {1, 2, 3}, 1: {3, 4, 5, 6}}
    """
    new_map = {}
    for bit, atom_pairs in info.items():
        unique_atoms = set()
        # Collapse atom pairs into unique atoms
        for atom1, atom2 in atom_pairs:
            unique_atoms.add(atom1)
            unique_atoms.add(atom2)

        new_map[bit] = unique_atoms

    return new_map


def get_bit_atom_map(
        mol: Chem.Mol, fp_generator: rdFingerprintGenerator
) -> dict[int, set[int]]:
    """
    Map bit features to atoms in a molecule. Currently only supports AtomPair fingerprints.

    :param mol: RDKit molecule object of the molecule to map
    :param fp_generator: RDKIt fingerprint generator object used to generate the features
    :return: Dictionary with bit features as keys and sets of atom indices as values associated with the bit
    """
    ao = AllChem.AdditionalOutput()
    ao.CollectBitInfoMap()
    _ = fp_generator.GetFingerprint(mol, additionalOutput=ao)
    # This contains the mapping of bits to atoms. It must be further processed (see below)
    info = ao.GetBitInfoMap()

    # We need to see what type of fingerprint generator we have in order to correctly format the output
    fp_gen_type = fp_generator.GetOptions().__str__()

    # Process bit-atom map
    if "AtomPair" in fp_gen_type:
        return _process_atom_pair_bits(info=info)
