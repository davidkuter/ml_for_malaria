from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ml_for_malaria.train.checkpoints import to_jsonable


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

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    auc = float(roc_auc_score(y_true, y_proba))
    cm = confusion_matrix(y_true, y_pred)

    def _avg(name: str) -> dict:
        row = report.get(name, {})
        return {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": float(row.get("support", 0.0)),
        }

    per_class = {}
    for key, value in report.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        per_class[str(key)] = {
            "precision": float(value["precision"]),
            "recall": float(value["recall"]),
            "f1": float(value["f1-score"]),
            "support": float(value["support"]),
        }

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "per_class": per_class,
        "macro": _avg("macro avg"),
        "weighted": _avg("weighted avg"),
        "confusion_matrix": cm.tolist(),
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


def report_to_markdown(report: dict) -> str:
    metrics = report["test_metrics"]
    per_class = metrics.get("per_class", {})
    class_names = sorted(per_class.keys(), key=lambda x: (x != "0", x != "1", x))

    header = "| metric | " + " | ".join(class_names + ["macro", "weighted"]) + " |"
    sep = "| --- | " + " | ".join("---" for _ in class_names + ["macro", "weighted"]) + " |"

    def _row(metric: str) -> str:
        cells = [metric]
        for name in class_names:
            cells.append(f"{per_class[name][metric]:.4f}")
        cells.append(f"{metrics['macro'][metric]:.4f}")
        cells.append(f"{metrics['weighted'][metric]:.4f}")
        return "| " + " | ".join(cells) + " |"

    fp_lines = [
        "| fingerprint | cv_auc | n_estimators |",
        "| --- | --- | --- |",
    ]
    comparison = report.get("fingerprint_comparison", {})
    for name, info in sorted(
        comparison.items(), key=lambda item: item[1].get("cv_auc", 0), reverse=True
    ):
        fp_lines.append(
            f"| {name} | {info.get('cv_auc', 0):.4f} | {info.get('n_estimators', '')} |"
        )

    arch_line = []
    if report.get("architecture"):
        arch_line = [f"- architecture: `{report['architecture']}`"]

    lines = [
        "# Training report",
        "",
        *arch_line,
        f"- split: `{report['split']}`",
        f"- seed: {report['seed']}",
        f"- test_size: {report['test_size']}",
        f"- n_train: {report['n_train']}",
        f"- n_test: {report['n_test']}",
        f"- best_fingerprint: `{report['best_fingerprint']}`",
        "",
        f"## Test metrics (threshold {metrics.get('threshold', 0.5)})",
        "",
        f"- accuracy: {metrics['accuracy']:.4f}",
        f"- roc_auc: {metrics['roc_auc']:.4f}",
        "",
        header,
        sep,
        _row("precision"),
        _row("recall"),
        _row("f1"),
        _row("support"),
        "",
        "## Fingerprint comparison (CV AUC)",
        "",
        *fp_lines,
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict, json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
