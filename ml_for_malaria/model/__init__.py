from ml_for_malaria.model.knn_classifier import KNNFingerprintClassifier
from ml_for_malaria.model.load import load_classifier
from ml_for_malaria.model.logistic_classifier import LogisticFingerprintClassifier
from ml_for_malaria.model.rf_classifier import RFFingerprintClassifier
from ml_for_malaria.model.xgb_classifier import ARCHITECTURE, XGBFingerprintClassifier

__all__ = [
    "ARCHITECTURE",
    "KNNFingerprintClassifier",
    "LogisticFingerprintClassifier",
    "RFFingerprintClassifier",
    "XGBFingerprintClassifier",
    "load_classifier",
]
