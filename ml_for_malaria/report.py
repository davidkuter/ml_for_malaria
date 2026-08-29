from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ml_for_malaria.runs.checkpoints import RunCheckpointer
from ml_for_malaria.schemas import (
    Architecture,
    ClassLabel,
    ComparisonReport,
    ComparisonRow,
    ComparisonTable,
    EvalMetrics,
    FingerprintComparison,
    FingerprintScore,
    MetricRow,
    MetricsTable,
    ModelMeta,
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
_NONE_IDENTIFIER = "none"


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
    """Test-set metrics at a fixed probability threshold, plus ROC-AUC and PR-AUC."""
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
        pr_auc=float(average_precision_score(y_true, y_proba)),
        per_class=per_class,
        macro=_metric_row(clf, SklearnReport.MACRO_AVG),
        weighted=_metric_row(clf, SklearnReport.WEIGHTED_AVG),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def class_f1(metrics: EvalMetrics, label: str) -> float:
    row = metrics.per_class.get(label)
    return float(row.f1) if row is not None else 0.0


def build_report(
    *,
    split: str,
    seed: int,
    test_size: float,
    n_train: int,
    n_test: int,
    test_metrics: EvalMetrics,
    architecture: str | None = None,
    best_fingerprint: str | None = None,
    fingerprint_comparison: dict | None = None,
    charge_method: str | None = None,
    pretrained_name: str | None = None,
) -> TrainingReport:
    return TrainingReport(
        split=split,
        seed=seed,
        test_size=test_size,
        n_train=n_train,
        n_test=n_test,
        test_metrics=test_metrics,
        architecture=architecture,
        best_fingerprint=best_fingerprint,
        fingerprint_comparison=fingerprint_comparison or {},
        charge_method=charge_method,
        pretrained_name=pretrained_name,
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
                TrainingReport.charge_method,
                TrainingReport.pretrained_name,
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
                EvalMetrics.pr_auc: metrics.pr_auc,
            }
        )
        .rename_axis(MetricsTable.metric)
        .reset_index(name=MetricsTable.value)
    )


def _class_names(metrics: EvalMetrics) -> list[str]:
    names = [str(name) for name in metrics.per_class]
    preferred = [
        name for name in (ClassLabel.INACTIVE, ClassLabel.ACTIVE) if name in names
    ]
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
        FingerprintComparison.roc_auc,
        FingerprintComparison.pr_auc,
        FingerprintComparison.accuracy,
        FingerprintComparison.f1_0,
        FingerprintComparison.f1_1,
        FingerprintComparison.weighted_f1,
    ]
    if not comparison:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            FingerprintComparison.fingerprint: name,
            FingerprintComparison.cv_auc: score.cv_auc,
            FingerprintComparison.n_estimators: score.n_estimators,
            FingerprintComparison.roc_auc: (
                score.test_metrics.roc_auc if score.test_metrics is not None else np.nan
            ),
            FingerprintComparison.pr_auc: (
                score.test_metrics.pr_auc if score.test_metrics is not None else np.nan
            ),
            FingerprintComparison.accuracy: (
                score.test_metrics.accuracy
                if score.test_metrics is not None
                else np.nan
            ),
            FingerprintComparison.f1_0: (
                class_f1(score.test_metrics, ClassLabel.INACTIVE)
                if score.test_metrics is not None
                else np.nan
            ),
            FingerprintComparison.f1_1: (
                class_f1(score.test_metrics, ClassLabel.ACTIVE)
                if score.test_metrics is not None
                else np.nan
            ),
            FingerprintComparison.weighted_f1: (
                score.test_metrics.weighted.f1
                if score.test_metrics is not None
                else np.nan
            ),
        }
        for name, score in comparison.items()
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame[FingerprintComparison.n_estimators] = frame[
        FingerprintComparison.n_estimators
    ].astype("Int64")
    return frame.sort_values(FingerprintComparison.cv_auc, ascending=False).reset_index(
        drop=True
    )


def _fingerprint_detail_sections(comparison: dict[str, FingerprintScore]) -> list[str]:
    if not comparison:
        return []
    ordered = _fingerprint_frame(comparison)[FingerprintComparison.fingerprint].tolist()
    parts: list[str] = []
    for name in ordered:
        metrics = comparison[name].test_metrics
        if metrics is None:
            continue
        parts.extend(
            [
                f"### {name}",
                "",
                "#### Per class",
                "",
                _markdown_table(_per_class_frame(metrics), index=True),
                "",
                "#### Confusion matrix",
                "",
                _markdown_table(_confusion_frame(metrics), index=True, floatfmt=None),
                "",
            ]
        )
    return parts


def report_to_markdown(report: TrainingReport) -> str:
    metrics = report.test_metrics
    sections = [
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
    ]
    if report.fingerprint_comparison:
        sections.extend(
            [
                "## Fingerprint comparison",
                "",
                _markdown_table(_fingerprint_frame(report.fingerprint_comparison)),
                "",
                *_fingerprint_detail_sections(report.fingerprint_comparison),
            ]
        )
    return "\n".join(sections)


def write_report(report: TrainingReport, json_path: Path, md_path: Path) -> None:
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")


def _run_identifier(
    report: TrainingReport, meta: ModelMeta | None, charge_method: str | None
) -> str:
    if report.architecture == Architecture.XGBOOST:
        return report.best_fingerprint or _NONE_IDENTIFIER
    if report.architecture == Architecture.CHEMBERTA:
        name = report.pretrained_name
        if name is None and meta is not None:
            name = meta.pretrained_name
        return name or _NONE_IDENTIFIER
    if charge_method:
        return charge_method
    return _NONE_IDENTIFIER


def _metrics_row(
    report: TrainingReport,
    *,
    charge_method: str | None,
    cleaned_hash: str | None,
    outdir: Path,
    meta: ModelMeta | None,
) -> ComparisonRow:
    metrics = report.test_metrics
    return ComparisonRow(
        architecture=report.architecture or _NONE_IDENTIFIER,
        identifier=_run_identifier(report, meta, charge_method),
        n_train=report.n_train,
        n_test=report.n_test,
        roc_auc=metrics.roc_auc,
        pr_auc=metrics.pr_auc,
        accuracy=metrics.accuracy,
        f1_0=class_f1(metrics, ClassLabel.INACTIVE),
        f1_1=class_f1(metrics, ClassLabel.ACTIVE),
        weighted_f1=metrics.weighted.f1,
        charge_method=charge_method,
        split=report.split,
        seed=report.seed,
        test_size=report.test_size,
        cleaned_hash=cleaned_hash,
        outdir=str(outdir),
    )


def _mismatch_warnings(rows: list[ComparisonRow]) -> list[str]:
    if len(rows) < 2:
        return []
    frame = pd.DataFrame([row.model_dump() for row in rows])
    warnings: list[str] = []
    for column in (
        ComparisonRow.split,
        ComparisonRow.seed,
        ComparisonRow.test_size,
        ComparisonRow.cleaned_hash,
    ):
        values = frame[column]
        unique = values.dropna().unique()
        if len(unique) > 1:
            warnings.append(f"Mismatched {column}: {sorted(map(str, unique))}")
            logger.warning(warnings[-1])
    return warnings


def _comparison_frame(rows: list[ComparisonRow]) -> pd.DataFrame:
    columns = [
        ComparisonTable.architecture,
        ComparisonTable.identifier,
        ComparisonTable.n_train,
        ComparisonTable.n_test,
        ComparisonTable.roc_auc,
        ComparisonTable.pr_auc,
        ComparisonTable.accuracy,
        ComparisonTable.f1_0,
        ComparisonTable.f1_1,
        ComparisonTable.weighted_f1,
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([row.model_dump() for row in rows])
    return frame.loc[:, columns]


def comparison_to_markdown(report: ComparisonReport) -> str:
    sections = [
        "# Architecture comparison",
        "",
        "## Test metrics",
        "",
        _markdown_table(_comparison_frame(report.rows)),
        "",
    ]
    if report.warnings:
        sections.extend(
            [
                "## Warnings",
                "",
                *[f"- {warning}" for warning in report.warnings],
                "",
            ]
        )
    return "\n".join(sections)


def write_comparison_report(
    run_dirs: list[str | Path],
    out_path: str | Path,
) -> ComparisonReport:
    """Compare held-out test metrics across completed run directories."""
    rows: list[ComparisonRow] = []
    for raw in run_dirs:
        outdir = Path(raw)
        ckpt = RunCheckpointer(outdir)
        if not ckpt.report_json_path.exists():
            raise FileNotFoundError(f"No report.json in {outdir}")
        report = TrainingReport.model_validate(ckpt.load_json(ckpt.report_json_path))
        config = ckpt.load_config()
        meta = (
            ModelMeta.model_validate(ckpt.load_json(ckpt.meta_path))
            if ckpt.meta_path.exists()
            else None
        )
        charge_method = report.charge_method
        if charge_method is None and config is not None:
            charge_method = config.charge_method
        if charge_method is None and meta is not None:
            charge_method = meta.charge_method
        cleaned_hash = config.cleaned_hash if config is not None else None
        rows.append(
            _metrics_row(
                report,
                charge_method=charge_method,
                cleaned_hash=cleaned_hash,
                outdir=outdir,
                meta=meta,
            )
        )
    comparison = ComparisonReport(rows=rows, warnings=_mismatch_warnings(rows))
    out_path = Path(out_path)
    json_path = out_path.with_suffix(".json")
    md_path = out_path.with_suffix(".md")
    json_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(comparison_to_markdown(comparison), encoding="utf-8")
    return comparison
