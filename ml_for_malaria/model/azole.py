import pandas as pd
from loguru import logger
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBClassifier

from ml_for_malaria.model.base import shap_feature_importance
from ml_for_malaria.train.featurization import (
    featurize_smiles,
    sanitize_smiles,
)


class AzoleModel:
    """
    XGBClassifier model for predicting the haemozoin inhibitory activity of azole-containing compounds

    """

    def __init__(self):
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
        """
        Load a pre-trained XGBoost Classifier model from a file
        :param model_path: Path to the model file
        :return:
        """
        self.model = XGBClassifier(**self.params)
        self.model.load_model(model_path)
        logger.debug(f"Parameters: {self.model.get_xgb_params()}")

    def featurize(self, smiles: list[str], sanitize: bool = False) -> pd.DataFrame:
        """
        Featurize a list of SMILES strings using a RDKit fingerprint generator. Optionally sanitise the SMILES strings
        :param smiles: List of SMILES strings to featurize
        :param sanitize: Boolean flag, if True sanitise the SMILES strings
        :return: DataFrame containing the features
        """
        return featurize_smiles(
            smiles=smiles, fp_generator=self.feature_generator, sanitize=sanitize
        )

    def predict(self, smiles: list[str]) -> pd.DataFrame:
        """
        Predict the probability of a list of SMILES strings being active

        :param smiles: List of SMILES strings to predict
        :return:
        """
        if self.model is None:
            raise RuntimeError(
                f'No model loaded. Please use ".load_model(model_path) before predicting"'
            )

        # Prepare data for prediction
        df = pd.DataFrame(
            zip(smiles, [sanitize_smiles(smiles=smi, as_mol=False) for smi in smiles]),
            columns=["INPUT_SMILES", "SMILES"],
        )
        prepped_smiles = list(df["SMILES"].dropna().unique())
        # df = df.dropna(subset=['SMILES'])
        # df = df.drop_duplicates(subset=['SMILES'])
        features = self.featurize(smiles=prepped_smiles, sanitize=False)

        # Predict
        results = self.model.predict_proba(features)

        # Reformat results
        # "results" is a list of tuples with probabilities for each label in our case 0 and 1.
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

    def get_feature_importance(
            self, smiles: str, img_out: str | None = None
    ) -> pd.DataFrame:
        """
        Get the feature importance of a single SMILES string. Optionally save an image of the feature importance
        :param smiles: SMILES string to get feature importance for
        :param img_out: (optional) Path to save the image of the feature importance
        :return: DataFrame containing the feature importance
        """
        if isinstance(smiles, str) is False:
            raise TypeError(
                "Feature importance can only be performed on a single SMILES"
            )

        return shap_feature_importance(
            smiles=smiles,
            model=self.model,
            feature_generator=self.feature_generator,
            img_out=img_out,
        )
