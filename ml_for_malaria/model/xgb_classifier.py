from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier

from ml_for_malaria.interpretation.shap import shap_feature_importance
from ml_for_malaria.schemas import CleanedTrainingData, ModelMeta, Predictions
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

    def __init__(self, model: XGBClassifier, feature_generator, metadata: ModelMeta):
        self.model = model
        self.feature_generator = feature_generator
        self.metadata = metadata

    @classmethod
    def load(
        cls, outdir: str | Path, fingerprint: str | None = None
    ) -> XGBFingerprintClassifier:
        """Load an XGBoost run. ``fingerprint`` selects ``models/{fingerprint}/``."""
        ckpt = RunCheckpointer(outdir)
        if fingerprint is None:
            meta_path = ckpt.meta_path
            model_path = ckpt.model_path
        else:
            meta_path = ckpt.fingerprint_meta_path(fingerprint)
            model_path = ckpt.fingerprint_model_path(fingerprint)
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"No saved model in {ckpt.outdir}. Expected {model_path.name} "
                f"and {meta_path.name}"
            )
        metadata = ModelMeta.model_validate(ckpt.load_json(meta_path))
        if metadata.architecture != ARCHITECTURE:
            raise ValueError(
                f"Run directory {ckpt.outdir} was trained with "
                f"architecture={metadata.architecture!r}, "
                f"not {ARCHITECTURE!r}. Load it with the matching classifier class."
            )
        generator = get_fingerprint_generator(
            metadata.fingerprint, fp_size=int(metadata.fp_size)
        )
        model = XGBClassifier()
        model.load_model(str(model_path))
        logger.debug(
            f"Loaded XGBoost model from {model_path} with {metadata.fingerprint}"
        )
        return cls(model=model, feature_generator=generator, metadata=metadata)

    def featurize(
        self, smiles: list[str] | pd.Series, sanitize: bool = False
    ) -> pd.DataFrame:
        return featurize_smiles(
            smiles=smiles,
            fp_generator=self.feature_generator,
            sanitize=sanitize,
        )

    def predict(self, smiles: list[str] | pd.Series) -> pd.DataFrame:
        """Predict the probability of the positive class for each input SMILES."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        input_smiles = CleanedTrainingData.INPUT_SMILES
        sanitized = CleanedTrainingData.SMILES
        probability = Predictions.PROBABILITY
        output_smiles = Predictions.SMILES

        df = pd.DataFrame({input_smiles: pd.Series(smiles, dtype="object")})
        df[sanitized] = df[input_smiles].map(
            lambda smi: sanitize_smiles(smi, as_mol=False)
        )
        unique_smiles = df[sanitized].dropna().drop_duplicates()
        features = self.featurize(smiles=unique_smiles, sanitize=False)
        output = df.drop(columns=sanitized).rename(
            columns={input_smiles: output_smiles}
        )
        if features.empty:
            return Predictions.validate(output.assign(**{probability: np.nan}))

        probabilities = pd.Series(
            self.model.predict_proba(features)[:, -1],
            index=features.index,
            name=probability,
        )
        output[probability] = df[sanitized].map(probabilities)
        return Predictions.validate(output)

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
