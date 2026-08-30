from __future__ import annotations

from ml_for_malaria.model.sklearn_fingerprint import SklearnFingerprintClassifier
from ml_for_malaria.schemas import Architecture

ARCHITECTURE = Architecture.LOGISTIC


class LogisticFingerprintClassifier(SklearnFingerprintClassifier):
    """L2-regularized logistic regression on binary fingerprints."""

    architecture = ARCHITECTURE
