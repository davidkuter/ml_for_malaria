from __future__ import annotations

import pandas as pd

from ml_for_malaria.model.sklearn_fingerprint import SklearnFingerprintClassifier
from ml_for_malaria.schemas import Architecture

ARCHITECTURE = Architecture.KNN


class KNNFingerprintClassifier(SklearnFingerprintClassifier):
    """Tanimoto (binary Jaccard) k-NN fingerprint classifier."""

    architecture = ARCHITECTURE

    @staticmethod
    def features_for_model(features: pd.DataFrame) -> pd.DataFrame:
        return features.astype(bool)
