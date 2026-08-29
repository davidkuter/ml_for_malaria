import pandas as pd
import shap
from loguru import logger
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps

from ml_for_malaria.schemas import AtomShapWeights, ShapValues
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
            records = [
                {"atom": atom, "shap": float(shap_values.loc[bit, smiles])}
                for bit, atoms in bit_map.items()
                if bit in shap_values.index and shap_values.loc[bit, smiles] != 0.0
                for atom in atoms
            ]
            n_atoms = mol.GetNumAtoms()
            if records:
                weights = (
                    AtomShapWeights.validate(pd.DataFrame(records))
                    .groupby("atom")["shap"]
                    .median()
                    .reindex(range(n_atoms), fill_value=0.0)
                )
            else:
                weights = pd.Series(0.0, index=range(n_atoms))
            scale = float(weights.abs().max())
            if scale:
                weights = (weights / scale).round(3)
            fig = SimilarityMaps.GetSimilarityMapFromWeights(
                mol, weights.sort_index().to_dict()
            )
            fig.savefig(img_out, bbox_inches="tight")

    return ShapValues.validate(shap_values.sort_values(by=[smiles]))
