from pathlib import Path

import numpy as np

from ml_for_malaria.report_plots import (
    headline_method_scores,
    write_tukey_hsd_plot,
)
from ml_for_malaria.schemas import (
    Architecture,
    ComparisonReport,
    ComparisonRow,
    Split,
)


def _row(
    *,
    architecture: str,
    identifier: str,
    seed: int,
    roc: float,
) -> ComparisonRow:
    return ComparisonRow(
        architecture=architecture,
        identifier=identifier,
        n_train=100,
        n_test=20,
        roc_auc=roc,
        pr_auc=roc,
        accuracy=0.5,
        f1_0=0.5,
        f1_1=0.5,
        weighted_f1=0.5,
        split=Split.SCAFFOLD,
        seed=seed,
        test_size=0.2,
        outdir=f"/tmp/{architecture}/{seed}",
    )


def test_headline_methods_pick_best_fingerprint():
    rows = []
    for seed in range(3):
        rows.append(
            _row(
                architecture=Architecture.RANDOM_FOREST,
                identifier="Morgan2Bits",
                seed=seed,
                roc=0.9 + 0.01 * seed,
            )
        )
        rows.append(
            _row(
                architecture=Architecture.RANDOM_FOREST,
                identifier="AtomPair",
                seed=seed,
                roc=0.7 + 0.01 * seed,
            )
        )
        rows.append(
            _row(
                architecture=Architecture.MONROE,
                identifier="TabPFN",
                seed=seed,
                roc=0.85 + 0.01 * seed,
            )
        )
    report = ComparisonReport(rows=rows, aggregates=[])
    series = headline_method_scores(report)
    assert "RF Morgan2Bits" in series
    assert "RF AtomPair" not in series
    assert "Monroe + TabPFN" in series
    assert len(series["RF Morgan2Bits"]) == 3


def test_write_tukey_hsd_plot(tmp_path: Path):
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(8):
        rows.append(
            _row(
                architecture=Architecture.RANDOM_FOREST,
                identifier="Morgan2Bits",
                seed=seed,
                roc=float(0.92 + rng.normal(0, 0.01)),
            )
        )
        rows.append(
            _row(
                architecture=Architecture.CHEMPROP,
                identifier="none",
                seed=seed,
                roc=float(0.75 + rng.normal(0, 0.02)),
            )
        )
        rows.append(
            _row(
                architecture=Architecture.MONROE,
                identifier="TabPFN",
                seed=seed,
                roc=float(0.91 + rng.normal(0, 0.01)),
            )
        )
    report = ComparisonReport(rows=rows, aggregates=[])
    out = write_tukey_hsd_plot(report, tmp_path / "tukey.png")
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 1000
