"""Pandera schemas for tables and Pydantic models for JSON artifacts.

Raw caller input is not validated here. Class attributes on these models are the
field names to use instead of string literals.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict, Field


class JsonModel(BaseModel):
    """Pydantic model whose class attributes are JSON field names."""

    __test__ = False
    model_config = ConfigDict(extra="ignore")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for name in cls.model_fields:
            setattr(cls, name, name)


class Architecture(StrEnum):
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    CHEMPROP = "chemprop"
    CHEMBERTA = "chemberta"


class SklearnClassWeight(StrEnum):
    BALANCED = "balanced"
    BALANCED_SUBSAMPLE = "balanced_subsample"


class RFMaxFeatures(StrEnum):
    SQRT = "sqrt"
    LOG2 = "log2"


class ChargeMethod(StrEnum):
    NAGL = "nagl"
    GASTEIGER = "gasteiger"


class PretrainedCheckpoint(StrEnum):
    CHEMBERTA_77M_MTR = "DeepChem/ChemBERTa-77M-MTR"
    TINY_TEST = "tiny-test"


class ClassLabel(StrEnum):
    """String labels used in sklearn classification reports and per-class tables."""

    INACTIVE = "0"
    ACTIVE = "1"


class MetricRow(JsonModel):
    precision: float
    recall: float
    f1: float
    support: float


class EvalMetrics(JsonModel):
    threshold: float
    accuracy: float
    roc_auc: float
    pr_auc: float
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
    tree_method: str = "hist"


class RandomForestParams(JsonModel):
    n_estimators: int
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str = RFMaxFeatures.SQRT
    class_weight: str | None = SklearnClassWeight.BALANCED
    n_jobs: int = 1
    random_state: int


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
    cv_auc: float | None = None
    n_estimators: int
    params: dict[str, Any] = Field(default_factory=dict)
    test_metrics: EvalMetrics | None = None


class SplitIndices(JsonModel):
    train_idx: list[int]
    test_idx: list[int]


class RunConfig(JsonModel):
    input_hash: str
    split: str
    seed: int
    test_size: float
    architecture: str
    fp_size: int | None = None
    max_evals: int | None = None
    fingerprints: list[str] | None = None
    cleaned_hash: str | None = None
    pretrained_name: str | None = None
    max_epochs: int | None = None
    batch_size: int | None = None
    charge_method: str | None = None
    freeze_encoder: bool | None = None
    hidden_size: int | None = None


class ModelMeta(JsonModel):
    architecture: str
    fingerprint: str = ""
    fp_size: int = 0
    n_estimators: int = 0
    params: dict[str, Any] = Field(default_factory=dict)
    pretrained_name: str | None = None
    charge_method: str | None = None
    freeze_encoder: bool | None = None


class TrainingReport(JsonModel):
    split: str
    seed: int
    test_size: float
    n_train: int
    n_test: int
    test_metrics: EvalMetrics
    architecture: str | None = None
    best_fingerprint: str | None = None
    fingerprint_comparison: dict[str, FingerprintScore] = Field(default_factory=dict)
    charge_method: str | None = None
    pretrained_name: str | None = None


class ComparisonRow(JsonModel):
    architecture: str
    identifier: str
    n_train: int
    n_test: int
    roc_auc: float
    pr_auc: float
    accuracy: float
    f1_0: float
    f1_1: float
    weighted_f1: float
    charge_method: str | None = None
    split: str
    seed: int
    test_size: float
    cleaned_hash: str | None = None
    outdir: str


class ComparisonAggregate(JsonModel):
    architecture: str
    identifier: str
    split: str
    charge_method: str | None = None
    n_seeds: int
    n_train: float
    n_test: float
    roc_auc_mean: float
    roc_auc_std: float | None = None
    pr_auc_mean: float
    pr_auc_std: float | None = None
    accuracy_mean: float
    accuracy_std: float | None = None
    f1_0_mean: float
    f1_0_std: float | None = None
    f1_1_mean: float
    f1_1_std: float | None = None
    weighted_f1_mean: float
    weighted_f1_std: float | None = None


class ComparisonReport(JsonModel):
    rows: list[ComparisonRow]
    aggregates: list[ComparisonAggregate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def empty_frame(schema: type[pa.DataFrameModel]) -> pd.DataFrame:
    """Schema columns with no rows. Callers check ``df.empty``, not ``None``."""
    columns = list(schema.to_schema().columns)
    return schema.validate(pd.DataFrame(columns=columns))


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


class AtomChargeCache(pa.DataFrameModel):
    """SMILES-keyed partial charges shared across Chemprop seeds."""

    SMILES: Series[str] = pa.Field(unique=True, nullable=False)
    charges: Series[object] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = "filter"


class FingerprintComparison(pa.DataFrameModel):
    """Per-fingerprint CV and held-out test results used in the training report."""

    fingerprint: Series[str]
    cv_auc: Series[float] = pa.Field(nullable=True)
    n_estimators: Series[int]
    roc_auc: Series[float]
    pr_auc: Series[float]
    accuracy: Series[float]
    f1_0: Series[float]
    f1_1: Series[float]
    weighted_f1: Series[float]

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


class ComparisonTable(pa.DataFrameModel):
    architecture: Series[str]
    identifier: Series[str]
    split: Series[str]
    n_train: Series[int]
    n_test: Series[int]
    roc_auc: Series[float]
    pr_auc: Series[float]
    accuracy: Series[float]
    f1_0: Series[float]
    f1_1: Series[float]
    weighted_f1: Series[float]

    class Config:
        coerce = True
        strict = False
