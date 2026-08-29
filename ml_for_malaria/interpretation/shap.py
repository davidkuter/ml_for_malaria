import numpy as np
import pandas as pd
import shap
from loguru import logger
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps

from ml_for_malaria.train.featurization import (
    featurize_smiles,
    get_bit_atom_map,
    sanitize_smiles,
)


def shap_feature_importance(
    smiles: str,
    model,
    feature_generator,
    img_out: str | None = None,
) -> pd.DataFrame:
    """SHAP values for a single SMILES string, optionally drawn on the standardized mol."""
    if not isinstance(smiles, str):
        raise TypeError("Feature importance can only be performed on a single SMILES")

    mol = sanitize_smiles(smiles, as_mol=True)
    if mol is None:
        raise ValueError(f"Could not sanitize SMILES for SHAP: {smiles!r}")
    std_smiles = Chem.MolToSmiles(mol)

    feats = featurize_smiles(
        smiles=[std_smiles], fp_generator=feature_generator, sanitize=False
    )
    if feats.empty:
        raise ValueError(f"Could not featurize SMILES for SHAP: {smiles!r}")

    explainer = shap.Explainer(model)
    shap_values_raw = explainer.shap_values(feats)
    shap_values = pd.DataFrame(shap_values_raw)
    shap_values.index.name = "FEATURE"
    shap_values = shap_values.T
    shap_values = shap_values.rename(columns={0: smiles})

    if img_out:
        bit_map = get_bit_atom_map(mol=mol, fp_generator=feature_generator)
        if bit_map is None:
            logger.error(
                "Atom highlighting is not supported for this fingerprint type; "
                "skipping SHAP image"
            )
        else:
            atom_shap = {atom.GetIdx(): [] for atom in mol.GetAtoms()}
            for bit, atoms in bit_map.items():
                if bit not in shap_values.index:
                    continue
                val = shap_values.loc[bit, smiles]
                if val != 0.0:
                    for atom in atoms:
                        if atom in atom_shap:
                            atom_shap[atom].append(val)

            weights = {}
            for atom, values in atom_shap.items():
                weights[atom] = 0 if len(values) == 0 else float(np.median(values))

            max_contrib = max((abs(val) for val in weights.values()), default=0.0)
            if max_contrib == 0:
                weights = {atom: 0.0 for atom in weights}
            else:
                weights = {
                    atom: round(weight / max_contrib, 3)
                    for atom, weight in weights.items()
                }

            weights = dict(sorted(weights.items()))
            fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights)
            fig.savefig(img_out, bbox_inches="tight")

    return shap_values.sort_values(by=[smiles])
