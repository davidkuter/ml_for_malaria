"""Pandera schemas for internal tables (not raw caller input)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class CleanedTrainingData(pa.DataFrameModel):
    """Sanitized unique SMILES with binary labels after ``clean_training_data``."""

    SMILES: Series[str] = pa.Field(unique=True, nullable=False)
    INPUT_SMILES: Series[str] = pa.Field(nullable=False)
    LABEL: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        coerce = True
        strict = "filter"


class FingerprintFeatures(pa.DataFrameModel):
    """Bit-vector fingerprints; columns are bit indices with values in {0, 1}."""

    class Config:
        strict = False

    @pa.dataframe_check
    def columns_are_bit_indices(cls, df: pd.DataFrame) -> bool:
        if df.shape[1] == 0:
            return True
        return bool(
            pd.to_numeric(pd.Index(df.columns).astype(str), errors="coerce")
            .notna()
            .all()
        )

    @pa.dataframe_check
    def bits_are_binary(cls, df: pd.DataFrame) -> bool:
        if df.shape[1] == 0:
            return True
        return bool(np.isin(df.to_numpy(), (0, 1)).all())


class Predictions(pa.DataFrameModel):
    """Positive-class probabilities aligned with the original input SMILES."""

    SMILES: Series[str] = pa.Field(nullable=False)
    PROBABILITY: Series[float] = pa.Field(nullable=True, ge=0.0, le=1.0)

    class Config:
        coerce = True
        strict = True


class AtomShapWeights(pa.DataFrameModel):
    """Per-atom SHAP contributions before grouping onto the molecule."""

    atom: Series[int] = pa.Field(ge=0, nullable=False)
    shap: Series[float] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = True


class ShapValues(pa.DataFrameModel):
    """One column of SHAP values (named by SMILES) indexed by fingerprint bit."""

    class Config:
        coerce = True
        strict = False

    @pa.dataframe_check
    def single_numeric_column(cls, df: pd.DataFrame) -> bool:
        return df.shape[1] == 1 and bool(
            pd.api.types.is_numeric_dtype(df.iloc[:, 0])
        )
