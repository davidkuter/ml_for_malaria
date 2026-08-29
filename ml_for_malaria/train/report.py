from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ml_for_malaria.train.checkpoints import to_jsonable

_SKIP_REPORT_ROWS = ("accuracy", "macro avg", "weighted avg")
_METRIC_ROWS = ("precision", "recall", "f1", "support")


def _metric_row(frame: pd.DataFrame, name: str) -> dict[str, float]:
    if name not in frame.index:
        return {col: 0.0 for col in _METRIC_ROWS}
    return frame.loc[name, list(_METRIC_ROWS)].astype(float).to_dict()


def compute_test_metrics(
    y_true,
    y_pred,
    y_proba,
    threshold: float = 0.5,
) -> dict:
    """Test-set metrics at a fixed probability threshold, plus ROC-AUC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    clf = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).T.rename(columns={"f1-score": "f1"})
    per_class = (
        clf.drop(index=clf.index.intersection(_SKIP_REPORT_ROWS))
        .reindex(columns=list(_METRIC_ROWS))
        .astype(float)
        .rename(index=str)
        .to_dict(orient="index")
    )

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "per_class": per_class,
        "macro": _metric_row(clf, "macro avg"),
        "weighted": _metric_row(clf, "weighted avg"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def build_report(
    *,
    split: str,
    seed: int,
    test_size: float,
    n_train: int,
    n_test: int,
    best_fingerprint: str,
    fingerprint_comparison: dict,
    test_metrics: dict,
    architecture: str | None = None,
) -> dict:
    payload = {
        "split": split,
        "seed": seed,
        "test_size": test_size,
        "n_train": n_train,
        "n_test": n_test,
        "best_fingerprint": best_fingerprint,
        "fingerprint_comparison": fingerprint_comparison,
        "test_metrics": test_metrics,
    }
    if architecture is not None:
        payload["architecture"] = architecture
    return to_jsonable(payload)


def _markdown_table(
    frame: pd.DataFrame, *, index: bool = False, floatfmt: str | None = ".4f"
) -> str:
    if frame.empty:
        return "_No data._"
    kwargs: dict = {"index": index, "tablefmt": "github"}
    if floatfmt is not None:
        kwargs["floatfmt"] = floatfmt
    return frame.to_markdown(**kwargs)


def _run_settings_frame(report: dict) -> pd.DataFrame:
    settings = pd.Series(
        {
            "architecture": report.get("architecture"),
            "split": report["split"],
            "seed": report["seed"],
            "test_size": report["test_size"],
            "n_train": report["n_train"],
            "n_test": report["n_test"],
            "best_fingerprint": report["best_fingerprint"],
        }
    ).dropna()
    return settings.rename_axis("setting").reset_index(name="value")


def _summary_metrics_frame(metrics: dict) -> pd.DataFrame:
    return (
        pd.Series(
            {
                "threshold": metrics.get("threshold", 0.5),
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        .rename_axis("metric")
        .reset_index(name="value")
    )


def _class_names(metrics: dict) -> list[str]:
    names = [str(name) for name in metrics.get("per_class", {})]
    preferred = [name for name in ("0", "1") if name in names]
    return preferred + [name for name in names if name not in preferred]


def _per_class_frame(metrics: dict) -> pd.DataFrame:
    table = pd.DataFrame(metrics.get("per_class", {})).reindex(columns=_class_names(metrics))
    table["macro"] = pd.Series(metrics["macro"])
    table["weighted"] = pd.Series(metrics["weighted"])
    return table.reindex(_METRIC_ROWS).rename_axis("metric")


def _confusion_frame(metrics: dict) -> pd.DataFrame:
    matrix = pd.DataFrame(metrics["confusion_matrix"])
    labels = _class_names(metrics)
    if len(labels) == len(matrix):
        matrix.index = [f"true_{label}" for label in labels]
        matrix.columns = [f"pred_{label}" for label in labels]
    return matrix


def _fingerprint_frame(comparison: dict) -> pd.DataFrame:
    if not comparison:
        return pd.DataFrame(columns=["fingerprint", "cv_auc", "n_estimators"])
    frame = (
        pd.DataFrame.from_dict(comparison, orient="index")
        .rename_axis("fingerprint")
        .reset_index()
    )
    columns = [
        col for col in ("fingerprint", "cv_auc", "n_estimators") if col in frame.columns
    ]
    frame = frame.loc[:, columns]
    if "n_estimators" in frame.columns:
        frame["n_estimators"] = frame["n_estimators"].astype("Int64")
    if "cv_auc" in frame.columns:
        frame = frame.sort_values("cv_auc", ascending=False)
    return frame.reset_index(drop=True)


def report_to_markdown(report: dict) -> str:
    metrics = report["test_metrics"]
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
            _markdown_table(_fingerprint_frame(report.get("fingerprint_comparison", {}))),
            "",
        ]
    )


def write_report(report: dict, json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
