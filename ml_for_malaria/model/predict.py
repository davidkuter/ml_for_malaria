from __future__ import annotations

import pandas as pd

from ml_for_malaria.schemas import CleanedTrainingData, Predictions
from ml_for_malaria.train.featurization import sanitize_smiles


def prepare_predict_smiles(
    smiles: list[str] | pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Align raw input SMILES with unique sanitized structures for batch inference."""
    input_smiles = CleanedTrainingData.INPUT_SMILES
    sanitized = CleanedTrainingData.SMILES
    frame = pd.DataFrame({input_smiles: pd.Series(smiles, dtype="object")})
    frame[sanitized] = frame[input_smiles].map(
        lambda smi: sanitize_smiles(smi, as_mol=False)
    )
    output = frame.drop(columns=sanitized).rename(
        columns={input_smiles: Predictions.SMILES}
    )
    unique = frame[sanitized].dropna().drop_duplicates()
    return output, frame[sanitized], unique
