import datamol as dm
import numpy as np
import pandas as pd

from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator


def sanitise_smiles(smiles: str, as_mol=False) -> str | Chem.Mol | None:
    if isinstance(smiles, str):
        mol = dm.to_mol(smiles)
        mol = dm.standardize_mol(mol)
        if as_mol is True:
            return mol
        else:
            return Chem.MolToSmiles(mol)
    else:
        logger.warning(f'"{smiles}" failed sanitisation')
        return None


def featurise_smiles(
        smiles: list[str], fp_generator: rdFingerprintGenerator, sanitise: bool = False
) -> pd.DataFrame:
    logger.info(f"Featurising {len(smiles)} SMILES")

    # Generate features
    features = []
    for smi in smiles:
        if sanitise:
            mol = sanitise_smiles(smi, as_mol=True)
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

    :param info:
    :return:
    """
    new_map = {}
    for bit, atom_pairs in info.items():
        unique_atoms = set()
        for atom1, atom2 in atom_pairs:
            unique_atoms.add(atom1)
            unique_atoms.add(atom2)

        new_map[bit] = unique_atoms

    return new_map


def get_bit_atom_map(mol: Chem.Mol, fp_generator: rdFingerprintGenerator):
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
