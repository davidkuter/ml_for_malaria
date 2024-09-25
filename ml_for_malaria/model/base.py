import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps

from ml_for_malaria.train.featurization import featurize_smiles, get_bit_atom_map


def shap_feature_importance(
        smiles: str,
        model,
        feature_generator: rdFingerprintGenerator,
        img_out: str | None = None,
) -> pd.DataFrame:
    """
    Get the feature importance of a single SMILES string for a XGBoost model.
    Optionally save an image of the feature importance.

    :param smiles: SMILES string to get feature importance for
    :param model: XGBoost model object
    :param feature_generator: RDKit fingerprint generator object
    :param img_out: (optional) Path to save the image of the feature importance
    :return: DataFrame containing the feature importance
    """
    if isinstance(smiles, str) is False:
        raise TypeError("Feature importance can only be performed on a single SMILES")

    # Featurize
    feats = featurize_smiles(
        smiles=[smiles], fp_generator=feature_generator, sanitize=True
    )

    # Perform SHAP feature importance
    explainer = shap.Explainer(model)
    shap_values_raw = explainer.shap_values(feats)
    shap_values = pd.DataFrame(shap_values_raw)
    shap_values.index.name = "FEATURE"
    shap_values = shap_values.T
    shap_values = shap_values.rename(columns={0: smiles})

    # Generate image
    if img_out:
        mol = Chem.MolFromSmiles(smiles)
        # Get info on what atoms are involved with a bit
        bit_map = get_bit_atom_map(mol=mol, fp_generator=feature_generator)

        # Initialize atom dict to ensure all atom indices are present in the dictionary
        atom_shap = {atom.GetIdx(): [] for atom in mol.GetAtoms()}

        # Add SHAP contributions of bits to atoms involved with those bits (as long as they are non-zero)
        for bit, atoms in bit_map.items():
            val = shap_values.loc[bit, smiles]
            if val != 0.0:
                for atom in atoms:
                    atom_shap[atom].append(val)

        # Get the median of SHAP contributions per atom
        weights = {}
        for atom, shaps in atom_shap.items():
            if len(shaps) == 0:
                weights[atom] = 0
            else:
                weights[atom] = np.median(shaps)

        # Normalize
        max_contrib = max([abs(val) for val in weights.values()])
        weights = {
            atom: round(weight / max_contrib, 3) for atom, weight in weights.items()
        }

        # Ensure atoms are sorted by index
        weights = dict(sorted(weights.items()))
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights)

        # Save image
        fig.savefig(img_out, bbox_inches="tight")

    return shap_values.sort_values(by=[smiles])
