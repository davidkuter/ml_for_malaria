import pandas as pd
import pytest
from pandera.errors import SchemaError

from ml_for_malaria.schemas import FingerprintFeatures, Predictions
from ml_for_malaria.train.featurization import clean_training_data


def test_clean_training_data_is_internal_schema():
    df = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCC"],
            "LABEL": [0, 1],
            "note": ["keep-out", "keep-out"],
        }
    )
    cleaned = clean_training_data(df)
    assert set(cleaned.columns) == {"SMILES", "INPUT_SMILES", "LABEL"}
    assert cleaned["LABEL"].tolist() == [0, 1]


def test_clean_training_data_rejects_non_binary_labels():
    df = pd.DataFrame({"SMILES": ["CCO", "CCC"], "LABEL": [0, 2]})
    with pytest.raises(SchemaError):
        clean_training_data(df)


def test_fingerprint_features_reject_non_binary_bits():
    features = pd.DataFrame([[0, 2]], columns=[0, 1])
    with pytest.raises(SchemaError):
        FingerprintFeatures.validate(features)


def test_predictions_reject_probability_outside_unit_interval():
    predictions = pd.DataFrame({"SMILES": ["CCO"], "PROBABILITY": [1.5]})
    with pytest.raises(SchemaError):
        Predictions.validate(predictions)
