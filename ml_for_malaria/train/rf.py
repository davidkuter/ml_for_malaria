from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from ml_for_malaria.chemistry.featurization import (
    DEFAULT_FP_SIZE,
    default_saved_fingerprint,
    get_fingerprint_generators,
)
from ml_for_malaria.model.rf_classifier import ARCHITECTURE, RFFingerprintClassifier
from ml_for_malaria.report import build_report, compute_test_metrics, write_report
from ml_for_malaria.runs.checkpoints import to_jsonable
from ml_for_malaria.runs.paths import resolve_run_dir
from ml_for_malaria.schemas import (
    CleanedTrainingData,
    EvalMetrics,
    FingerprintScore,
    ModelMeta,
    RandomForestParams,
    RunConfig,
    TrainingReport,
)
from ml_for_malaria.train.fingerprint_features import fingerprint_features_for_run
from ml_for_malaria.train.prepare import prepare_training_run

DEFAULT_N_ESTIMATORS = 200


@dataclass
class RFTrainResult:
    model: RandomForestClassifier
    classifier: RFFingerprintClassifier
    report: TrainingReport
    outdir: Path


def default_rf_params(seed: int, n_estimators: int = DEFAULT_N_ESTIMATORS) -> RandomForestParams:
    """Fixed forest recipe (no hyperparameter search)."""
    return RandomForestParams(
        n_estimators=int(n_estimators),
        random_state=int(seed),
    )


def _fit_fingerprint_model(
    features: pd.DataFrame,
    y: pd.Series,
    train_idx: list[int],
    test_idx: list[int],
    params: RandomForestParams,
) -> tuple[RandomForestClassifier, EvalMetrics]:
    clf = RandomForestClassifier(**params.model_dump())
    clf.fit(features.iloc[train_idx], y.iloc[train_idx])
    y_pred = clf.predict(features.iloc[test_idx])
    y_proba = clf.predict_proba(features.iloc[test_idx])[:, 1]
    test_metrics = compute_test_metrics(y.iloc[test_idx], y_pred, y_proba)
    saved = RandomForestClassifier(**params.model_dump())
    saved.fit(features, y)
    return saved, test_metrics


def train_rf_classifier(
    df: pd.DataFrame,
    outdir: str | Path,
    split: str = "random",
    seed: int = 42,
    test_size: float = 0.2,
    force: bool = False,
    fp_size: int = DEFAULT_FP_SIZE,
    fingerprints: list[str] | None = None,
) -> RFTrainResult:
    """Train a random-forest fingerprint classifier with checkpointed artifacts.

    ``df`` must contain SMILES and LABEL (0/1) columns.
    ``outdir`` is the parent runs directory; artifacts go in
    ``{outdir}/{arch}_{split}/seed_{seed}/``. Every fingerprint is scored on the
    frozen test split. The default ``model.joblib`` is ``DEFAULT_FINGERPRINT``
    when that generator was trained. There is no hyperparameter search.
    """
    parent = Path(outdir)
    outdir = resolve_run_dir(parent, ARCHITECTURE, split, seed=seed)
    prepared = prepare_training_run(
        df,
        outdir,
        split=split,
        seed=seed,
        test_size=test_size,
        force=force,
    )
    ckpt = prepared.ckpt
    available = get_fingerprint_generators(fp_size=fp_size)
    selected = list(available) if fingerprints is None else list(fingerprints)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown fingerprint(s) {unknown}. Supported: {sorted(available)}"
        )

    expected = RunConfig(
        input_hash=prepared.input_hash,
        split=split,
        seed=seed,
        test_size=test_size,
        fp_size=fp_size,
        fingerprints=selected,
        architecture=ARCHITECTURE,
        cleaned_hash=prepared.cleaned_hash,
    )

    if ckpt.run_complete(prepared.stored, expected):
        logger.info(f"Reusing completed run in {ckpt.outdir}")
        classifier = RFFingerprintClassifier.load(ckpt.outdir)
        report = TrainingReport.model_validate(ckpt.load_json(ckpt.report_json_path))
        return RFTrainResult(
            model=classifier.model,
            classifier=classifier,
            report=report,
            outdir=ckpt.outdir,
        )

    cleaned = prepared.cleaned
    train_idx = prepared.train_idx
    test_idx = prepared.test_idx
    generators = {name: available[name] for name in selected}
    y = cleaned[CleanedTrainingData.LABEL]
    fingerprint_comparison: dict[str, FingerprintScore] = {}
    stored = prepared.stored
    params = default_rf_params(seed)

    for fp_name, generator in generators.items():
        logger.info(f"Evaluating {fp_name}")
        features = fingerprint_features_for_run(
            ckpt=ckpt,
            parent=parent,
            fp_name=fp_name,
            generator=generator,
            cleaned=cleaned,
            cleaned_hash=prepared.cleaned_hash,
            fp_size=fp_size,
            stored=stored,
        )
        logger.info(f"Validating {fp_name} on held-out test split")
        forest, test_metrics = _fit_fingerprint_model(
            features, y, train_idx, test_idx, params
        )
        logger.info(f"{fp_name}: test ROC-AUC={test_metrics.roc_auc:.4f}")
        fingerprint_comparison[fp_name] = FingerprintScore(
            cv_auc=None,
            n_estimators=params.n_estimators,
            params=params.model_dump(),
            test_metrics=test_metrics,
        )
        metadata = ModelMeta(
            architecture=ARCHITECTURE,
            fingerprint=fp_name,
            fp_size=fp_size,
            params=to_jsonable(params.model_dump()),
            n_estimators=int(params.n_estimators),
        )
        ckpt.fingerprint_dir(fp_name).mkdir(parents=True, exist_ok=True)
        joblib.dump(forest, ckpt.fingerprint_sklearn_path(fp_name))
        ckpt.save_json(ckpt.fingerprint_meta_path(fp_name), metadata)

    best_fingerprint = default_saved_fingerprint(selected)
    logger.info(f"Default fingerprint artifact: {best_fingerprint}")
    best = fingerprint_comparison[best_fingerprint]
    shutil.copy2(
        ckpt.fingerprint_sklearn_path(best_fingerprint),
        ckpt.sklearn_model_path,
    )
    shutil.copy2(
        ckpt.fingerprint_meta_path(best_fingerprint),
        ckpt.meta_path,
    )

    report = build_report(
        split=split,
        seed=seed,
        test_size=test_size,
        n_train=len(train_idx),
        n_test=len(test_idx),
        best_fingerprint=best_fingerprint,
        fingerprint_comparison=fingerprint_comparison,
        test_metrics=best.test_metrics,
        architecture=ARCHITECTURE,
    )
    write_report(report, ckpt.report_json_path, ckpt.report_md_path)
    ckpt.save_config(expected)

    classifier = RFFingerprintClassifier.load(ckpt.outdir)
    return RFTrainResult(
        model=classifier.model,
        classifier=classifier,
        report=report,
        outdir=ckpt.outdir,
    )
