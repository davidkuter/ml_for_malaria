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
    FINGERPRINT_ARCHITECTURES,
    Architecture,
    ClassLabel,
    ComparisonAggregate,
    ComparisonHpoDelta,
    ComparisonReport,
    ComparisonRow,
    ComparisonSplitReference,
    ComparisonTable,
    ComparisonYscrambleDelta,
    EvalMetrics,
    FingerprintComparison,
    FingerprintScore,
    MetricRow,
    MetricsTable,
    ModelMeta,
    SettingsTable,
    Split,
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
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            "y_true, y_pred, and y_proba must have the same length "
            f"(got {y_true.shape[0]}, {y_pred.shape[0]}, {y_proba.shape[0]})"
        )

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
    max_evals: int | None = None,
    yscramble: bool = False,
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
        max_evals=max_evals,
        yscramble=yscramble,
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
                TrainingReport.max_evals,
            )
        }
    ).dropna()
    if report.yscramble:
        settings[TrainingReport.yscramble] = True
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
    sort_key = FingerprintComparison.cv_auc
    if frame[sort_key].isna().all():
        sort_key = FingerprintComparison.roc_auc
    return frame.sort_values(sort_key, ascending=False).reset_index(drop=True)


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
    ]
    if report.max_evals and report.max_evals > 0:
        sections.extend(
            [
                (
                    "TPE hyperparameter search used **train folds only** "
                    f"(`max_evals={report.max_evals}`). "
                    "The held-out test split was not used to choose parameters."
                ),
                "",
            ]
        )
    if report.yscramble:
        sections.extend(
            [
                (
                    "Train labels were **y-scrambled** (permuted among train "
                    "compounds only). Held-out test labels are the real assay labels."
                ),
                "",
            ]
        )
    sections.extend(
        [
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
    )
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
    if report.architecture in FINGERPRINT_ARCHITECTURES:
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
    hpo: bool,
    yscramble: bool,
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
        hpo=hpo,
        yscramble=yscramble,
    )


def _report_is_hpo(report: TrainingReport) -> bool:
    return bool(report.max_evals and report.max_evals > 0)


def _report_is_yscramble(report: TrainingReport) -> bool:
    return bool(report.yscramble)


def _mismatch_warnings(rows: list[ComparisonRow]) -> list[str]:
    if len(rows) < 2:
        return []
    frame = pd.DataFrame([row.model_dump() for row in rows])
    warnings: list[str] = []
    for column in (ComparisonRow.test_size, ComparisonRow.cleaned_hash):
        values = frame[column]
        unique = values.dropna().unique()
        if len(unique) > 1:
            warnings.append(f"Mismatched {column}: {sorted(map(str, unique))}")
            logger.warning(warnings[-1])
    return warnings


def _sample_std(values: pd.Series) -> float | None:
    if len(values) < 2:
        return None
    return float(values.std(ddof=1))


def _format_mean_std(mean: float, std: float | None) -> str:
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _aggregate_rows(rows: list[ComparisonRow]) -> list[ComparisonAggregate]:
    if not rows:
        return []
    frame = pd.DataFrame([row.model_dump() for row in rows])
    group_cols = [
        ComparisonRow.architecture,
        ComparisonRow.identifier,
        ComparisonRow.split,
        ComparisonRow.charge_method,
        ComparisonRow.hpo,
        ComparisonRow.yscramble,
    ]
    aggregates: list[ComparisonAggregate] = []
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for keys, group in grouped:
        architecture, identifier, split, charge_method, hpo, yscramble = keys
        if pd.isna(charge_method):
            charge_method = None
        aggregates.append(
            ComparisonAggregate(
                architecture=architecture,
                identifier=identifier,
                split=split,
                charge_method=charge_method,
                hpo=bool(hpo),
                yscramble=bool(yscramble),
                n_seeds=int(group[ComparisonRow.seed].nunique()),
                n_train=float(group[ComparisonRow.n_train].mean()),
                n_test=float(group[ComparisonRow.n_test].mean()),
                roc_auc_mean=float(group[ComparisonRow.roc_auc].mean()),
                roc_auc_std=_sample_std(group[ComparisonRow.roc_auc]),
                pr_auc_mean=float(group[ComparisonRow.pr_auc].mean()),
                pr_auc_std=_sample_std(group[ComparisonRow.pr_auc]),
                accuracy_mean=float(group[ComparisonRow.accuracy].mean()),
                accuracy_std=_sample_std(group[ComparisonRow.accuracy]),
                f1_0_mean=float(group[ComparisonRow.f1_0].mean()),
                f1_0_std=_sample_std(group[ComparisonRow.f1_0]),
                f1_1_mean=float(group[ComparisonRow.f1_1].mean()),
                f1_1_std=_sample_std(group[ComparisonRow.f1_1]),
                weighted_f1_mean=float(group[ComparisonRow.weighted_f1].mean()),
                weighted_f1_std=_sample_std(group[ComparisonRow.weighted_f1]),
            )
        )
    return aggregates


def _aggregate_frame(aggregates: list[ComparisonAggregate]) -> pd.DataFrame:
    columns = [
        ComparisonTable.architecture,
        ComparisonTable.identifier,
        ComparisonTable.split,
        ComparisonRow.hpo,
        ComparisonRow.yscramble,
        ComparisonAggregate.n_seeds,
        ComparisonTable.n_train,
        ComparisonTable.n_test,
        ComparisonTable.roc_auc,
        ComparisonTable.pr_auc,
        ComparisonTable.accuracy,
        ComparisonTable.f1_0,
        ComparisonTable.f1_1,
        ComparisonTable.weighted_f1,
    ]
    if not aggregates:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            ComparisonTable.architecture: item.architecture,
            ComparisonTable.identifier: item.identifier,
            ComparisonTable.split: item.split,
            ComparisonRow.hpo: item.hpo,
            ComparisonRow.yscramble: item.yscramble,
            ComparisonAggregate.n_seeds: item.n_seeds,
            ComparisonTable.n_train: item.n_train,
            ComparisonTable.n_test: item.n_test,
            ComparisonTable.roc_auc: _format_mean_std(
                item.roc_auc_mean, item.roc_auc_std
            ),
            ComparisonTable.pr_auc: _format_mean_std(item.pr_auc_mean, item.pr_auc_std),
            ComparisonTable.accuracy: _format_mean_std(
                item.accuracy_mean, item.accuracy_std
            ),
            ComparisonTable.f1_0: _format_mean_std(item.f1_0_mean, item.f1_0_std),
            ComparisonTable.f1_1: _format_mean_std(item.f1_1_mean, item.f1_1_std),
            ComparisonTable.weighted_f1: _format_mean_std(
                item.weighted_f1_mean, item.weighted_f1_std
            ),
        }
        for item in aggregates
    ]
    return pd.DataFrame(rows, columns=columns)


def _per_seed_frame(rows: list[ComparisonRow]) -> pd.DataFrame:
    columns = [
        ComparisonTable.architecture,
        ComparisonTable.identifier,
        ComparisonTable.split,
        ComparisonRow.hpo,
        ComparisonRow.yscramble,
        ComparisonRow.seed,
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


def _format_delta(value: float) -> str:
    return f"{value:+.4f}"


def _hpo_deltas(aggregates: list[ComparisonAggregate]) -> list[ComparisonHpoDelta]:
    fixed = {
        (item.architecture, item.identifier, item.split, item.charge_method): item
        for item in aggregates
        if not item.hpo and not item.yscramble
    }
    deltas: list[ComparisonHpoDelta] = []
    for item in aggregates:
        if not item.hpo or item.yscramble:
            continue
        baseline = fixed.get(
            (item.architecture, item.identifier, item.split, item.charge_method)
        )
        if baseline is None:
            continue
        deltas.append(
            ComparisonHpoDelta(
                architecture=item.architecture,
                identifier=item.identifier,
                split=item.split,
                charge_method=item.charge_method,
                n_seeds=min(item.n_seeds, baseline.n_seeds),
                roc_auc_fixed=baseline.roc_auc_mean,
                roc_auc_hpo=item.roc_auc_mean,
                roc_auc_delta=item.roc_auc_mean - baseline.roc_auc_mean,
                pr_auc_fixed=baseline.pr_auc_mean,
                pr_auc_hpo=item.pr_auc_mean,
                pr_auc_delta=item.pr_auc_mean - baseline.pr_auc_mean,
                weighted_f1_fixed=baseline.weighted_f1_mean,
                weighted_f1_hpo=item.weighted_f1_mean,
                weighted_f1_delta=item.weighted_f1_mean - baseline.weighted_f1_mean,
            )
        )
    return deltas


def best_fixed_scaffold_identifier(
    aggregates: list[ComparisonAggregate],
    architecture: str,
    *,
    split: str = Split.SCAFFOLD,
) -> str:
    """Fingerprint (or charge) with the highest fixed-recipe mean scaffold ROC."""
    rows = [
        item
        for item in aggregates
        if item.architecture == architecture
        and item.split == split
        and not item.hpo
        and not item.yscramble
    ]
    if not rows:
        raise ValueError(
            f"No fixed-recipe {split} rows for architecture {architecture!r}"
        )
    return max(rows, key=lambda item: item.roc_auc_mean).identifier


def _hpo_delta_frame(deltas: list[ComparisonHpoDelta]) -> pd.DataFrame:
    columns = [
        ComparisonTable.architecture,
        ComparisonTable.identifier,
        ComparisonTable.split,
        ComparisonHpoDelta.n_seeds,
        ComparisonHpoDelta.roc_auc_fixed,
        ComparisonHpoDelta.roc_auc_hpo,
        ComparisonHpoDelta.roc_auc_delta,
        ComparisonHpoDelta.pr_auc_delta,
        ComparisonHpoDelta.weighted_f1_delta,
    ]
    if not deltas:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            ComparisonTable.architecture: item.architecture,
            ComparisonTable.identifier: item.identifier,
            ComparisonTable.split: item.split,
            ComparisonHpoDelta.n_seeds: item.n_seeds,
            ComparisonHpoDelta.roc_auc_fixed: f"{item.roc_auc_fixed:.4f}",
            ComparisonHpoDelta.roc_auc_hpo: f"{item.roc_auc_hpo:.4f}",
            ComparisonHpoDelta.roc_auc_delta: _format_delta(item.roc_auc_delta),
            ComparisonHpoDelta.pr_auc_delta: _format_delta(item.pr_auc_delta),
            ComparisonHpoDelta.weighted_f1_delta: _format_delta(item.weighted_f1_delta),
        }
        for item in deltas
    ]
    return pd.DataFrame(rows, columns=columns)


def _yscramble_deltas(
    aggregates: list[ComparisonAggregate],
) -> list[ComparisonYscrambleDelta]:
    real = {
        (item.architecture, item.identifier, item.split, item.charge_method): item
        for item in aggregates
        if not item.hpo and not item.yscramble
    }
    deltas: list[ComparisonYscrambleDelta] = []
    for item in aggregates:
        if not item.yscramble or item.hpo:
            continue
        baseline = real.get(
            (item.architecture, item.identifier, item.split, item.charge_method)
        )
        if baseline is None:
            continue
        deltas.append(
            ComparisonYscrambleDelta(
                architecture=item.architecture,
                identifier=item.identifier,
                split=item.split,
                charge_method=item.charge_method,
                n_seeds=min(item.n_seeds, baseline.n_seeds),
                roc_auc_real=baseline.roc_auc_mean,
                roc_auc_scramble=item.roc_auc_mean,
                roc_auc_delta=item.roc_auc_mean - baseline.roc_auc_mean,
                pr_auc_real=baseline.pr_auc_mean,
                pr_auc_scramble=item.pr_auc_mean,
                pr_auc_delta=item.pr_auc_mean - baseline.pr_auc_mean,
                weighted_f1_real=baseline.weighted_f1_mean,
                weighted_f1_scramble=item.weighted_f1_mean,
                weighted_f1_delta=item.weighted_f1_mean - baseline.weighted_f1_mean,
            )
        )
    return deltas


def _yscramble_delta_frame(
    deltas: list[ComparisonYscrambleDelta],
) -> pd.DataFrame:
    columns = [
        ComparisonTable.architecture,
        ComparisonTable.identifier,
        ComparisonTable.split,
        ComparisonYscrambleDelta.n_seeds,
        ComparisonYscrambleDelta.roc_auc_real,
        ComparisonYscrambleDelta.roc_auc_scramble,
        ComparisonYscrambleDelta.roc_auc_delta,
        ComparisonYscrambleDelta.pr_auc_delta,
        ComparisonYscrambleDelta.weighted_f1_delta,
    ]
    if not deltas:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            ComparisonTable.architecture: item.architecture,
            ComparisonTable.identifier: item.identifier,
            ComparisonTable.split: item.split,
            ComparisonYscrambleDelta.n_seeds: item.n_seeds,
            ComparisonYscrambleDelta.roc_auc_real: f"{item.roc_auc_real:.4f}",
            ComparisonYscrambleDelta.roc_auc_scramble: f"{item.roc_auc_scramble:.4f}",
            ComparisonYscrambleDelta.roc_auc_delta: _format_delta(item.roc_auc_delta),
            ComparisonYscrambleDelta.pr_auc_delta: _format_delta(item.pr_auc_delta),
            ComparisonYscrambleDelta.weighted_f1_delta: _format_delta(
                item.weighted_f1_delta
            ),
        }
        for item in deltas
    ]
    return pd.DataFrame(rows, columns=columns)


def _split_references(
    aggregates: list[ComparisonAggregate],
) -> list[ComparisonSplitReference]:
    """Each random-split row vs the best fixed-recipe scaffold identifier."""
    refs: list[ComparisonSplitReference] = []
    random_rows = [
        item
        for item in aggregates
        if item.split == Split.RANDOM and not item.hpo and not item.yscramble
    ]
    architectures = {item.architecture for item in random_rows}
    for architecture in sorted(architectures):
        try:
            reference_id = best_fixed_scaffold_identifier(aggregates, architecture)
        except ValueError:
            continue
        reference = next(
            (
                item
                for item in aggregates
                if item.architecture == architecture
                and item.identifier == reference_id
                and item.split == Split.SCAFFOLD
                and not item.hpo
                and not item.yscramble
            ),
            None,
        )
        if reference is None:
            continue
        for item in random_rows:
            if item.architecture != architecture:
                continue
            refs.append(
                ComparisonSplitReference(
                    architecture=item.architecture,
                    identifier=item.identifier,
                    split=item.split,
                    reference_identifier=reference.identifier,
                    reference_split=reference.split,
                    n_seeds=min(item.n_seeds, reference.n_seeds),
                    n_test=item.n_test,
                    n_test_reference=reference.n_test,
                    roc_auc=item.roc_auc_mean,
                    roc_auc_reference=reference.roc_auc_mean,
                    roc_auc_delta=item.roc_auc_mean - reference.roc_auc_mean,
                    pr_auc=item.pr_auc_mean,
                    pr_auc_reference=reference.pr_auc_mean,
                    pr_auc_delta=item.pr_auc_mean - reference.pr_auc_mean,
                    weighted_f1=item.weighted_f1_mean,
                    weighted_f1_reference=reference.weighted_f1_mean,
                    weighted_f1_delta=item.weighted_f1_mean - reference.weighted_f1_mean,
                )
            )
    return refs


def _split_reference_frame(
    refs: list[ComparisonSplitReference],
) -> pd.DataFrame:
    columns = [
        ComparisonTable.architecture,
        ComparisonTable.identifier,
        ComparisonSplitReference.reference_identifier,
        ComparisonSplitReference.n_test,
        ComparisonSplitReference.n_test_reference,
        ComparisonTable.roc_auc,
        ComparisonSplitReference.roc_auc_reference,
        ComparisonSplitReference.roc_auc_delta,
        ComparisonSplitReference.pr_auc_delta,
        ComparisonSplitReference.weighted_f1_delta,
    ]
    if not refs:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            ComparisonTable.architecture: item.architecture,
            ComparisonTable.identifier: item.identifier,
            ComparisonSplitReference.reference_identifier: item.reference_identifier,
            ComparisonSplitReference.n_test: f"{item.n_test:.1f}",
            ComparisonSplitReference.n_test_reference: f"{item.n_test_reference:.1f}",
            ComparisonTable.roc_auc: f"{item.roc_auc:.4f}",
            ComparisonSplitReference.roc_auc_reference: f"{item.roc_auc_reference:.4f}",
            ComparisonSplitReference.roc_auc_delta: _format_delta(item.roc_auc_delta),
            ComparisonSplitReference.pr_auc_delta: _format_delta(item.pr_auc_delta),
            ComparisonSplitReference.weighted_f1_delta: _format_delta(
                item.weighted_f1_delta
            ),
        }
        for item in refs
    ]
    return pd.DataFrame(rows, columns=columns)


def comparison_to_markdown(report: ComparisonReport) -> str:
    sections = [
        "# Architecture comparison",
        "",
        "## Test metrics (mean ± std)",
        "",
        _markdown_table(_aggregate_frame(report.aggregates), floatfmt=None),
        "",
    ]
    if report.split_references:
        sections.extend(
            [
                "## Random split vs best scaffold (reference)",
                "",
                (
                    "Random split is an analogue-leakage diagnostic, not a ranking. "
                    "The reference is the best fixed-recipe **scaffold** identifier "
                    "for that architecture. Test size and class mix differ. "
                    "Δ is random-split mean minus scaffold reference mean."
                ),
                "",
                _markdown_table(
                    _split_reference_frame(report.split_references), floatfmt=None
                ),
                "",
            ]
        )
    if report.hpo_deltas:
        sections.extend(
            [
                "## HPO improvement vs fixed recipe",
                "",
                (
                    "TPE on **train folds only**. Fingerprint identity matches the "
                    "fixed-recipe row. Held-out test splits are unchanged. "
                    "Δ is HPO mean minus fixed-recipe mean."
                ),
                "",
                _markdown_table(_hpo_delta_frame(report.hpo_deltas), floatfmt=None),
                "",
            ]
        )
    if report.yscramble_deltas:
        sections.extend(
            [
                "## Y-scramble (train labels permuted)",
                "",
                (
                    "Train labels were shuffled among train compounds; "
                    "**test labels are real**. Same frozen splits as the matching "
                    "fixed-recipe row. Δ is scramble mean minus real-label mean. "
                    "A working model should drop toward chance (ROC ≈ 0.5)."
                ),
                "",
                _markdown_table(
                    _yscramble_delta_frame(report.yscramble_deltas), floatfmt=None
                ),
                "",
            ]
        )
    sections.extend(
        [
            "## Per-seed runs",
            "",
            _markdown_table(_per_seed_frame(report.rows)),
            "",
        ]
    )
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


def _rows_for_run(
    report: TrainingReport,
    *,
    charge_method: str | None,
    cleaned_hash: str | None,
    outdir: Path,
    meta: ModelMeta | None,
    hpo: bool,
    yscramble: bool,
) -> list[ComparisonRow]:
    if report.fingerprint_comparison:
        rows: list[ComparisonRow] = []
        for name, score in report.fingerprint_comparison.items():
            if score.test_metrics is None:
                continue
            per_fp = report.model_copy(
                update={
                    TrainingReport.best_fingerprint: name,
                    TrainingReport.test_metrics: score.test_metrics,
                }
            )
            rows.append(
                _metrics_row(
                    per_fp,
                    charge_method=charge_method,
                    cleaned_hash=cleaned_hash,
                    outdir=outdir,
                    meta=meta,
                    hpo=hpo,
                    yscramble=yscramble,
                )
            )
        if rows:
            return rows
    return [
        _metrics_row(
            report,
            charge_method=charge_method,
            cleaned_hash=cleaned_hash,
            outdir=outdir,
            meta=meta,
            hpo=hpo,
            yscramble=yscramble,
        )
    ]


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
        hpo = _report_is_hpo(report)
        if (
            not hpo
            and config is not None
            and config.max_evals is not None
            and config.max_evals > 0
        ):
            hpo = True
        yscramble = _report_is_yscramble(report)
        if (
            not yscramble
            and config is not None
            and config.yscramble is True
        ):
            yscramble = True
        rows.extend(
            _rows_for_run(
                report,
                charge_method=charge_method,
                cleaned_hash=cleaned_hash,
                outdir=outdir,
                meta=meta,
                hpo=hpo,
                yscramble=yscramble,
            )
        )
    aggregates = _aggregate_rows(rows)
    comparison = ComparisonReport(
        rows=rows,
        aggregates=aggregates,
        hpo_deltas=_hpo_deltas(aggregates),
        yscramble_deltas=_yscramble_deltas(aggregates),
        split_references=_split_references(aggregates),
        warnings=_mismatch_warnings(rows),
    )
    out_path = Path(out_path)
    json_path = out_path.with_suffix(".json")
    md_path = out_path.with_suffix(".md")
    json_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(comparison_to_markdown(comparison), encoding="utf-8")
    return comparison
