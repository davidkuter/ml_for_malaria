from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ml_for_malaria.schemas import (
    EvalMetrics,
    FingerprintComparison,
    FingerprintScore,
    MetricRow,
    MetricsTable,
    SettingsTable,
    TrainingReport,
)

class SklearnReport(StrEnum):
    """Keys from ``sklearn.metrics.classification_report(..., output_dict=True)``."""

    ACCURACY = "accuracy"
    MACRO_AVG = "macro avg"
    WEIGHTED_AVG = "weighted avg"
    F1_SCORE = "f1-score"


_METRIC_ROWS = tuple(MetricRow.model_fields)


def _metric_row(frame: pd.DataFrame, name: str) -> MetricRow:
    if name not in frame.index:
        return MetricRow(precision=0.0, recall=0.0, f1=0.0, support=0.0)
    values = frame.loc[name, list(_METRIC_ROWS)].astype(float)
    return MetricRow.model_validate(values.to_dict())


def compute_test_metrics(
    y_true,
    y_pred,
    y_proba,
    threshold: float = 0.5,
) -> EvalMetrics:
    """Test-set metrics at a fixed probability threshold, plus ROC-AUC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    skip_rows = (
        SklearnReport.ACCURACY,
        SklearnReport.MACRO_AVG,
        SklearnReport.WEIGHTED_AVG,
    )
    clf = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).T.rename(columns={SklearnReport.F1_SCORE: MetricRow.f1})
    per_class = {
        str(label): MetricRow.model_validate(row)
        for label, row in (
            clf.drop(index=clf.index.intersection(skip_rows))
            .reindex(columns=list(_METRIC_ROWS))
            .astype(float)
            .rename(index=str)
            .to_dict(orient="index")
            .items()
        )
    }

    return EvalMetrics(
        threshold=threshold,
        accuracy=float(accuracy_score(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        per_class=per_class,
        macro=_metric_row(clf, SklearnReport.MACRO_AVG),
        weighted=_metric_row(clf, SklearnReport.WEIGHTED_AVG),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def build_report(
    *,
    split: str,
    seed: int,
    test_size: float,
    n_train: int,
    n_test: int,
    best_fingerprint: str,
    fingerprint_comparison: dict,
    test_metrics: EvalMetrics,
    architecture: str | None = None,
) -> TrainingReport:
    return TrainingReport(
        split=split,
        seed=seed,
        test_size=test_size,
        n_train=n_train,
        n_test=n_test,
        best_fingerprint=best_fingerprint,
        fingerprint_comparison=fingerprint_comparison,
        test_metrics=test_metrics,
        architecture=architecture,
    )


def _markdown_table(
    frame: pd.DataFrame, *, index: bool = False, floatfmt: str | None = ".4f"
) -> str:
    if frame.empty:
        return "_No data._"
    kwargs: dict = {"index": index, "tablefmt": "github"}
    if floatfmt is not None:
        kwargs["floatfmt"] = floatfmt
    return frame.to_markdown(**kwargs)


def _run_settings_frame(report: TrainingReport) -> pd.DataFrame:
    settings = pd.Series(
        {
            name: getattr(report, name)
            for name in (
                TrainingReport.architecture,
                TrainingReport.split,
                TrainingReport.seed,
                TrainingReport.test_size,
                TrainingReport.n_train,
                TrainingReport.n_test,
                TrainingReport.best_fingerprint,
            )
        }
    ).dropna()
    return settings.rename_axis(SettingsTable.setting).reset_index(
        name=SettingsTable.value
    )


def _summary_metrics_frame(metrics: EvalMetrics) -> pd.DataFrame:
    return (
        pd.Series(
            {
                EvalMetrics.threshold: metrics.threshold,
                EvalMetrics.accuracy: metrics.accuracy,
                EvalMetrics.roc_auc: metrics.roc_auc,
            }
        )
        .rename_axis(MetricsTable.metric)
        .reset_index(name=MetricsTable.value)
    )


def _class_names(metrics: EvalMetrics) -> list[str]:
    names = [str(name) for name in metrics.per_class]
    preferred = [name for name in ("0", "1") if name in names]
    return preferred + [name for name in names if name not in preferred]


def _per_class_frame(metrics: EvalMetrics) -> pd.DataFrame:
    table = pd.DataFrame(
        {name: row.model_dump() for name, row in metrics.per_class.items()}
    ).reindex(columns=_class_names(metrics))
    table[EvalMetrics.macro] = pd.Series(metrics.macro.model_dump())
    table[EvalMetrics.weighted] = pd.Series(metrics.weighted.model_dump())
    return table.reindex(_METRIC_ROWS).rename_axis(MetricsTable.metric)


def _confusion_frame(metrics: EvalMetrics) -> pd.DataFrame:
    matrix = pd.DataFrame(metrics.confusion_matrix)
    labels = _class_names(metrics)
    if len(labels) == len(matrix):
        matrix.index = [f"true_{label}" for label in labels]
        matrix.columns = [f"pred_{label}" for label in labels]
    return matrix


def _fingerprint_frame(comparison: dict[str, FingerprintScore]) -> pd.DataFrame:
    columns = [
        FingerprintComparison.fingerprint,
        FingerprintComparison.cv_auc,
        FingerprintComparison.n_estimators,
    ]
    if not comparison:
        return pd.DataFrame(columns=columns)
    frame = (
        pd.DataFrame.from_dict(
            {name: score.model_dump() for name, score in comparison.items()},
            orient="index",
        )
        .rename_axis(FingerprintComparison.fingerprint)
        .reset_index()
    )
    present = [col for col in columns if col in frame.columns]
    frame = frame.loc[:, present]
    if FingerprintComparison.n_estimators in frame.columns:
        frame[FingerprintComparison.n_estimators] = frame[
            FingerprintComparison.n_estimators
        ].astype("Int64")
    if FingerprintComparison.cv_auc in frame.columns:
        frame = frame.sort_values(FingerprintComparison.cv_auc, ascending=False)
    return frame.reset_index(drop=True)


def report_to_markdown(report: TrainingReport) -> str:
    metrics = report.test_metrics
    return "\n".join(
        [
            "# Training report",
            "",
            "## Run",
            "",
            _markdown_table(_run_settings_frame(report)),
            "",
            "## Test metrics",
            "",
            _markdown_table(_summary_metrics_frame(metrics)),
            "",
            "### Per class",
            "",
            _markdown_table(_per_class_frame(metrics), index=True),
            "",
            "### Confusion matrix",
            "",
            _markdown_table(_confusion_frame(metrics), index=True, floatfmt=None),
            "",
            "## Fingerprint comparison (CV AUC)",
            "",
            _markdown_table(_fingerprint_frame(report.fingerprint_comparison)),
            "",
        ]
    )


def write_report(report: TrainingReport, json_path: Path, md_path: Path) -> None:
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
