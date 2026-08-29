from pathlib import Path

import pandas as pd
from loguru import logger
from xgboost import XGBClassifier

from ml_for_malaria.interpretation.shap import shap_feature_importance
from ml_for_malaria.train.checkpoints import RunCheckpointer
from ml_for_malaria.train.featurization import (
    featurize_smiles,
    get_fingerprint_generator,
    sanitize_smiles,
)

ARCHITECTURE = "xgboost"


class XGBFingerprintClassifier:
    """XGBoost fingerprint classifier restored from a training run directory."""

    architecture = ARCHITECTURE

    def __init__(self, model: XGBClassifier, feature_generator, metadata: dict):
        self.model = model
        self.feature_generator = feature_generator
        self.metadata = metadata

    @classmethod
    def load(cls, outdir: str | Path) -> "XGBFingerprintClassifier":
        """Load ``model.ubj`` and ``model_meta.json`` from an XGBoost training run."""
        ckpt = RunCheckpointer(outdir)
        if not ckpt.meta_path.exists() or not ckpt.model_path.exists():
            raise FileNotFoundError(
                f"No saved model in {ckpt.outdir}. Expected model.ubj and model_meta.json"
            )
        metadata = ckpt.load_json(ckpt.meta_path)
        saved_arch = metadata.get("architecture", ARCHITECTURE)
        if saved_arch != ARCHITECTURE:
            raise ValueError(
                f"Run directory {ckpt.outdir} was trained with architecture={saved_arch!r}, "
                f"not {ARCHITECTURE!r}. Load it with the matching classifier class."
            )
        generator = get_fingerprint_generator(
            metadata["fingerprint"], fp_size=int(metadata["fp_size"])
        )
        model = XGBClassifier()
        model.load_model(str(ckpt.model_path))
        logger.debug(f"Loaded XGBoost model from {ckpt.model_path} with {metadata['fingerprint']}")
        return cls(model=model, feature_generator=generator, metadata=metadata)

    def featurize(self, smiles: list[str], sanitize: bool = False) -> pd.DataFrame:
        return featurize_smiles(
            smiles=smiles, fp_generator=self.feature_generator, sanitize=sanitize
        )

    def predict(self, smiles: list[str]) -> pd.DataFrame:
        """Predict the probability of the positive class for each input SMILES."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        df = pd.DataFrame(
            {
                "INPUT_SMILES": smiles,
                "SMILES": [sanitize_smiles(smi, as_mol=False) for smi in smiles],
            }
        )
        prepped_smiles = list(df["SMILES"].dropna().unique())
        features = self.featurize(smiles=prepped_smiles, sanitize=False)
        if features.empty:
            df = df.drop(columns="SMILES")
            df = df.rename(columns={"INPUT_SMILES": "SMILES"})
            df["PROBABILITY"] = pd.NA
            return df

        results = self.model.predict_proba(features)
        df_results = pd.DataFrame(
            [(smi, float(prob[-1])) for smi, prob in zip(features.index, results)],
            columns=["SMILES", "PROBABILITY"],
        )
        df = df.merge(df_results, on="SMILES", how="left")
        df = df.drop(columns="SMILES")
        return df.rename(columns={"INPUT_SMILES": "SMILES"})

    def get_feature_importance(
        self, smiles: str, img_out: str | None = None
    ) -> pd.DataFrame:
        """SHAP feature importance for a single SMILES string."""
        if not isinstance(smiles, str):
            raise TypeError(
                "Feature importance can only be performed on a single SMILES"
            )
        if self.model is None:
            raise RuntimeError("No model loaded")
        return shap_feature_importance(
            smiles=smiles,
            model=self.model,
            feature_generator=self.feature_generator,
            img_out=img_out,
        )
