"""Walters-style Tukey HSD plots for architecture comparison reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import studentized_range, tukey_hsd

from ml_for_malaria.schemas import (
    FINGERPRINT_ARCHITECTURES,
    Architecture,
    ComparisonReport,
    ComparisonRow,
    EvalMetrics,
    Split,
)

_COLOR_BEST = "#1f77b4"
_COLOR_TIED = "#7f7f7f"
_COLOR_WORSE = "#d62728"
_NONE = "none"


def _method_label(architecture: str, identifier: str) -> str:
    if architecture == Architecture.MONROE:
        return "Monroe + TabPFN"
    if architecture == Architecture.CHEMELEON:
        return "ChemProp + CheMeleon"
    if architecture == Architecture.CHEMBERTA:
        return "ChemBERTa-77M-MTR"
    if architecture == Architecture.CHEMPROP:
        charge = identifier if identifier != _NONE else "none"
        return f"Chemprop ({charge})"
    if architecture == Architecture.RANDOM_FOREST:
        return f"RF {identifier}"
    if architecture == Architecture.XGBOOST:
        return f"XGB {identifier}"
    if architecture == Architecture.KNN:
        return f"k-NN {identifier}"
    if architecture == Architecture.LOGISTIC:
        return f"L2-logistic {identifier}"
    return f"{architecture} {identifier}".strip()


def _fixed_scaffold_rows(report: ComparisonReport) -> list[ComparisonRow]:
    return [
        row
        for row in report.rows
        if row.split == Split.SCAFFOLD and not row.hpo and not row.yscramble
    ]


def headline_method_scores(
    report: ComparisonReport,
    *,
    metric: str = EvalMetrics.roc_auc,
) -> dict[str, np.ndarray]:
    """One series per method: best fingerprint per tree arch; all DL / foundation arms.

    Values are per-seed held-out scores (same seeds, fixed scaffold recipe).
    """
    rows = _fixed_scaffold_rows(report)
    if not rows:
        return {}

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row.architecture, row.identifier)
        grouped.setdefault(key, []).append(float(getattr(row, metric)))

    means = {key: float(np.mean(vals)) for key, vals in grouped.items()}
    selected: dict[str, list[float]] = {}

    for architecture in FINGERPRINT_ARCHITECTURES:
        candidates = {key: mean for key, mean in means.items() if key[0] == architecture}
        if not candidates:
            continue
        best_key = max(candidates, key=candidates.get)
        selected[_method_label(*best_key)] = grouped[best_key]

    for architecture in (
        Architecture.MONROE,
        Architecture.CHEMELEON,
        Architecture.CHEMBERTA,
        Architecture.CHEMPROP,
    ):
        for key, vals in grouped.items():
            if key[0] != architecture:
                continue
            selected[_method_label(*key)] = vals

    return {label: np.asarray(vals, dtype=np.float64) for label, vals in selected.items()}


def _tukey_halfwidths(samples: list[np.ndarray], alpha: float = 0.05) -> np.ndarray:
    """Simultaneous CI half-widths (Hochberg / statsmodels equal-variance form)."""
    k = len(samples)
    counts = np.asarray([len(sample) for sample in samples], dtype=np.float64)
    df = float(counts.sum() - k)
    if df <= 0 or k < 2:
        return np.zeros(k, dtype=np.float64)
    sse = sum(float(((sample - sample.mean()) ** 2).sum()) for sample in samples)
    mse = sse / df
    q = float(studentized_range.ppf(1.0 - alpha, k, df))
    return (q / np.sqrt(2.0)) * np.sqrt(mse / counts)


def _status_vs_best(
    samples: list[np.ndarray],
    labels: list[str],
    *,
    higher_is_better: bool,
    alpha: float = 0.05,
) -> tuple[int, list[str]]:
    """Return best index and per-method status: best / tied / worse."""
    means = np.asarray([sample.mean() for sample in samples], dtype=np.float64)
    best_idx = int(np.argmax(means) if higher_is_better else np.argmin(means))
    if len(samples) < 2:
        return best_idx, ["best"] * len(samples)

    result = tukey_hsd(*samples)
    statuses: list[str] = []
    for idx, _label in enumerate(labels):
        if idx == best_idx:
            statuses.append("best")
            continue
        p_value = float(result.pvalue[best_idx, idx])
        statuses.append("tied" if p_value >= alpha else "worse")
    return best_idx, statuses


def write_tukey_hsd_plot(
    report: ComparisonReport,
    out_path: str | Path,
    *,
    metric: str = EvalMetrics.roc_auc,
    alpha: float = 0.05,
    title: str | None = None,
) -> Path | None:
    """Write a Walters-style Tukey HSD mean±CI plot (PNG).

    Blue = best mean; grey = not significantly different from best; red = worse.
    Dashed lines mark the best method's simultaneous CI.
    """
    series = headline_method_scores(report, metric=metric)
    if len(series) < 2:
        logger.warning("Need at least two methods for a Tukey HSD plot; skipping")
        return None

    labels = list(series.keys())
    samples = [series[label] for label in labels]
    means = np.asarray([sample.mean() for sample in samples], dtype=np.float64)
    halfwidths = _tukey_halfwidths(samples, alpha=alpha)
    higher_is_better = metric in (EvalMetrics.roc_auc, EvalMetrics.pr_auc, "weighted_f1")
    best_idx, statuses = _status_vs_best(
        samples, labels, higher_is_better=higher_is_better, alpha=alpha
    )

    order = np.argsort(means)
    labels = [labels[i] for i in order]
    means = means[order]
    halfwidths = halfwidths[order]
    statuses = [statuses[i] for i in order]
    best_idx = int(np.where(np.asarray(statuses) == "best")[0][0])

    colors = []
    for status in statuses:
        if status == "best":
            colors.append(_COLOR_BEST)
        elif status == "tied":
            colors.append(_COLOR_TIED)
        else:
            colors.append(_COLOR_WORSE)

    fig_h = max(4.5, 0.45 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    y = np.arange(len(labels))
    ax.errorbar(
        means,
        y,
        xerr=halfwidths,
        fmt="o",
        markersize=7,
        capsize=3,
        ecolor="0.35",
        elinewidth=1.2,
        markeredgecolor="0.2",
        markeredgewidth=0.6,
        color="none",
    )
    for yi, mean, color in zip(y, means, colors):
        ax.plot(mean, yi, "o", color=color, markersize=8, zorder=3)

    best_lo = means[best_idx] - halfwidths[best_idx]
    best_hi = means[best_idx] + halfwidths[best_idx]
    ax.axvline(best_lo, color=_COLOR_BEST, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(best_hi, color=_COLOR_BEST, linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    metric_name = {
        EvalMetrics.roc_auc: "ROC-AUC",
        EvalMetrics.pr_auc: "PR-AUC",
        "weighted_f1": "weighted F1 @0.5",
    }.get(metric, metric)
    ax.set_xlabel(f"Mean held-out {metric_name}")
    ax.set_title(
        title
        or (
            f"Tukey HSD on scaffold {metric_name} "
            f"(α={alpha:g}; blue=best, grey=tied, red=worse)"
        )
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    x_lo = float(np.min(means - halfwidths) - 0.02)
    x_hi = float(np.max(means + halfwidths) + 0.02)
    ax.set_xlim(x_lo, x_hi)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info(f"Wrote Tukey HSD plot to {out_path}")
    return out_path
