"""Pandera schemas for tables and Pydantic models for JSON artifacts.

Raw caller input is not validated here. Class attributes on these models are the
field names to use instead of string literals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict


class JsonModel(BaseModel):
    """Pydantic model whose class attributes are JSON field names."""

    __test__ = False
    model_config = ConfigDict(extra="ignore")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for name in cls.model_fields:
            setattr(cls, name, name)


class MetricRow(JsonModel):
    precision: float
    recall: float
    f1: float
    support: float


class EvalMetrics(JsonModel):
    threshold: float
    accuracy: float
    roc_auc: float
    per_class: dict[str, MetricRow]
    macro: MetricRow
    weighted: MetricRow
    confusion_matrix: list[list[int]]


class XGBParams(JsonModel):
    objective: str
    alpha: int
    gamma: float
    reg_lambda: float
    colsample_bytree: float
    min_child_weight: int
    max_depth: int
    learning_rate: float
    n_estimators: int
    random_state: int
    seed: int
    eval_metric: str


class HyperoptInjected(JsonModel):
    dtrain: Any


class HyperoptObjectiveResult(JsonModel):
    loss: float
    status: str
    n_estimators: int
    auc: float


class HyperoptResult(JsonModel):
    params: XGBParams
    cv_auc: float
    n_estimators: int


class FingerprintScore(JsonModel):
    cv_auc: float
    n_estimators: int
    params: dict[str, Any] = {}


class SplitIndices(JsonModel):
    train_idx: list[int]
    test_idx: list[int]


class RunConfig(JsonModel):
    input_hash: str
    split: str
    seed: int
    test_size: float
    fp_size: int
    max_evals: int
    fingerprints: list[str]
    architecture: str
    cleaned_hash: str | None = None


class ModelMeta(JsonModel):
    architecture: str
    fingerprint: str
    fp_size: int
    n_estimators: int = 0
    params: dict[str, Any] = {}


class TrainingReport(JsonModel):
    split: str
    seed: int
    test_size: float
    n_train: int
    n_test: int
    best_fingerprint: str
    fingerprint_comparison: dict[str, FingerprintScore]
    test_metrics: EvalMetrics
    architecture: str | None = None


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
        return df.shape[1] == 1 and bool(pd.api.types.is_numeric_dtype(df.iloc[:, 0]))


class FingerprintComparison(pa.DataFrameModel):
    """Per-fingerprint CV results used in the training report."""

    fingerprint: Series[str]
    cv_auc: Series[float]
    n_estimators: Series[int]

    class Config:
        coerce = True
        strict = False


class SettingsTable(pa.DataFrameModel):
    setting: Series[str]
    value: Series[str]

    class Config:
        coerce = True
        strict = False


class MetricsTable(pa.DataFrameModel):
    metric: Series[str]
    value: Series[float]

    class Config:
        coerce = True
        strict = False
