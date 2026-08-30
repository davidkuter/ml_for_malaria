from ml_for_malaria.train.knn import train_knn_classifier
from ml_for_malaria.train.logistic import train_logistic_classifier
from ml_for_malaria.train.rf import RFTrainResult, train_rf_classifier
from ml_for_malaria.train.sklearn_fingerprint import SklearnFingerprintTrainResult
from ml_for_malaria.train.xgb import XGBTrainResult, train_xgb_classifier

__all__ = [
    "RFTrainResult",
    "SklearnFingerprintTrainResult",
    "XGBTrainResult",
    "train_knn_classifier",
    "train_logistic_classifier",
    "train_rf_classifier",
    "train_xgb_classifier",
]
