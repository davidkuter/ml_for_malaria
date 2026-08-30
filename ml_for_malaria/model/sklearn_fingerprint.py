from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from ml_for_malaria.chemistry.featurization import (
    featurize_smiles,
    get_fingerprint_generator,
    sanitize_smiles,
)
from ml_for_malaria.runs.checkpoints import RunCheckpointer
from ml_for_malaria.schemas import CleanedTrainingData, ModelMeta, Predictions


class SklearnFingerprintClassifier:
    """Joblib sklearn fingerprint classifier restored from a training run directory."""

    architecture: str

    def __init__(self, model, feature_generator, metadata: ModelMeta):
        self.model = model
        self.feature_generator = feature_generator
        self.metadata = metadata

    @staticmethod
    def features_for_model(features: pd.DataFrame) -> pd.DataFrame:
        return features

    @classmethod
    def load(
        cls, outdir: str | Path, fingerprint: str | None = None
    ) -> SklearnFingerprintClassifier:
        """Load a sklearn fingerprint run. ``fingerprint`` selects ``models/{fingerprint}/``."""
        ckpt = RunCheckpointer(outdir)
        if fingerprint is None:
            meta_path = ckpt.meta_path
            model_path = ckpt.sklearn_model_path
        else:
            meta_path = ckpt.fingerprint_meta_path(fingerprint)
            model_path = ckpt.fingerprint_sklearn_path(fingerprint)
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"No saved model in {ckpt.outdir}. Expected {model_path.name} "
                f"and {meta_path.name}"
            )
        metadata = ModelMeta.model_validate(ckpt.load_json(meta_path))
        if metadata.architecture != cls.architecture:
            raise ValueError(
                f"Run directory {ckpt.outdir} was trained with "
                f"architecture={metadata.architecture!r}, "
                f"not {cls.architecture!r}. Load it with the matching classifier class."
            )
        generator = get_fingerprint_generator(
            metadata.fingerprint, fp_size=int(metadata.fp_size)
        )
        model = joblib.load(model_path)
        logger.debug(
            f"Loaded {cls.architecture} model from {model_path} "
            f"with {metadata.fingerprint}"
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
            self.model.predict_proba(self.features_for_model(features))[:, -1],
            index=features.index,
            name=probability,
        )
        output[probability] = df[sanitized].map(probabilities)
        return Predictions.validate(output)
