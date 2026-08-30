from __future__ import annotations

import pandas as pd

from ml_for_malaria.interpretation.shap import shap_feature_importance
from ml_for_malaria.model.sklearn_fingerprint import SklearnFingerprintClassifier
from ml_for_malaria.schemas import Architecture

ARCHITECTURE = Architecture.RANDOM_FOREST


class RFFingerprintClassifier(SklearnFingerprintClassifier):
    """Random forest fingerprint classifier restored from a training run directory."""

    architecture = ARCHITECTURE

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
