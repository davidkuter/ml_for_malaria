import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps
from xgboost import XGBClassifier

from ml_for_malaria.train.featurization import featurise_smiles, sanitise_smiles, get_bit_atom_map


class AzoleModel:
    def __init__(self):
        # self.feature_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
        self.feature_generator = rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=2048
        )
        self.model = None
        self.seed = 42

        self.params = {
            "objective": "binary:logistic",
            "alpha": 12,
            "colsample_bytree": 0.5351727658939963,
            "gamma": 1.0716563415167997,
            "learning_rate": 0.08749094940221283,
            "max_depth": 15,
            "min_child_weight": 6,
            "reg_lambda": 0.9470053856349937,
            "seed": self.seed,
        }

        self.stats = {
            "AUC": 0.9144897808569082,
            "precision": 0.7946589446589447,
            "recall": 0.7916666666666666,
            "f1-score": 0.7917872531995626,
            "support": 72.0,
        }

    def load_model(self, model_path: str):
        self.model = XGBClassifier(**self.params)
        self.model.load_model(model_path)
        logger.debug(f"Parameters: {self.model.get_xgb_params()}")

    def featurise(self, smiles: list[str], sanitise: bool = False) -> pd.DataFrame:
        return featurise_smiles(
            smiles=smiles, fp_generator=self.feature_generator, sanitise=sanitise
        )

    def predict(self, smiles: list[str]):
        if self.model is None:
            raise RuntimeError(
                f'No model loaded. Please use ".load_model(model_path) before predicting"'
            )

        # Prepare data for prediction
        df = pd.DataFrame(
            zip(smiles, [sanitise_smiles(smiles=smi, as_mol=False) for smi in smiles]),
            columns=["INPUT_SMILES", "SMILES"],
        )
        prepped_smiles = list(df["SMILES"].dropna().unique())
        # df = df.dropna(subset=['SMILES'])
        # df = df.drop_duplicates(subset=['SMILES'])
        features = self.featurise(smiles=prepped_smiles, sanitise=False)

        # Predict
        results = self.model.predict_proba(features)

        # Reformat results
        # results is a list of tuples with probabilities for each label in our case 0 and 1.
        # E.g. [[0.0394336  0.9605664 ],...]
        # We are interested in 1 (active) only so remove the result of the info
        df_results = pd.DataFrame(
            [(smi, prob[-1]) for smi, prob in zip(features.index, results)],
            columns=["SMILES", "PROBABILITY"],
        )
        df = df.merge(df_results, on="SMILES", how="left")
        df = df.drop(columns="SMILES")
        df = df.rename(columns={"INPUT_SMILES": "SMILES"})

        return df

    def get_feature_importance(self, smiles: str, img_out: str | None = None):
        if isinstance(smiles, str) is False:
            raise TypeError(
                "Feature importance can only be performed on a single SMILES"
            )

        # Featurise
        feats = self.featurise(smiles=[smiles], sanitise=True)

        # Perform SHAP feature importance
        explainer = shap.Explainer(self.model)
        shap_values_raw = explainer.shap_values(feats)
        shap_values = pd.DataFrame(shap_values_raw)
        shap_values.index.name = "FEATURE"
        shap_values = shap_values.T
        shap_values = shap_values.rename(columns={0: smiles})

        if img_out:
            mol = Chem.MolFromSmiles(smiles)
            # Get info on what atoms are involved with a bit
            bit_map = get_bit_atom_map(mol=mol, fp_generator=self.feature_generator)

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

            # Get predicted probability
            prob = str(round(self.model.predict_proba(feats)[0][1], 3))
            plt.title(f"PROBABILITY FOR ACTIVE: {prob}")

            # Save image
            fig.savefig(img_out, bbox_inches="tight")

        # feat_import = self.model.get_booster().get_score(importance_type='weight')
        # feat_import = pd.DataFrame.from_dict(feat_import, orient='index', columns=['WEIGHT'])
        # feat_import.index.name = 'SMILES'

        return shap_values.sort_values(by=[smiles])
