from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from ml_for_malaria.chemistry.featurization import (
    DEFAULT_FP_SIZE,
    default_saved_fingerprint,
    get_fingerprint_generators,
)
from ml_for_malaria.model.sklearn_fingerprint import SklearnFingerprintClassifier
from ml_for_malaria.report import build_report, compute_test_metrics, write_report
from ml_for_malaria.runs.checkpoints import to_jsonable
from ml_for_malaria.runs.paths import resolve_run_dir
from ml_for_malaria.schemas import (
    CleanedTrainingData,
    EvalMetrics,
    FingerprintScore,
    ModelMeta,
    RunConfig,
    TrainingReport,
)
from ml_for_malaria.train.fingerprint_features import fingerprint_features_for_run
from ml_for_malaria.train.prepare import prepare_training_run, scramble_train_labels

MakeEstimator = Callable[[BaseModel, int], Any]


@dataclass
class SklearnFingerprintTrainResult:
    model: Any
    classifier: SklearnFingerprintClassifier
    report: TrainingReport
    outdir: Path


def _fit_fingerprint_model(
    features: pd.DataFrame,
    y: pd.Series,
    train_idx: list[int],
    test_idx: list[int],
    *,
    params: BaseModel,
    make_estimator: MakeEstimator,
    features_for_model: Callable[[pd.DataFrame], pd.DataFrame],
    y_test: pd.Series | None = None,
) -> tuple[Any, EvalMetrics]:
    transformed = features_for_model(features)
    estimator = make_estimator(params, len(train_idx))
    estimator.fit(transformed.iloc[train_idx], y.iloc[train_idx])
    y_eval = y if y_test is None else y_test
    test_x = transformed.iloc[test_idx]
    y_pred = estimator.predict(test_x)
    y_proba = estimator.predict_proba(test_x)[:, 1]
    test_metrics = compute_test_metrics(y_eval.iloc[test_idx], y_pred, y_proba)
    saved = make_estimator(params, len(train_idx))
    saved.fit(transformed, y)
    return saved, test_metrics


def train_sklearn_fingerprint_classifier(
    df: pd.DataFrame,
    outdir: str | Path,
    *,
    architecture: str,
    classifier_cls: type[SklearnFingerprintClassifier],
    params: BaseModel,
    make_estimator: MakeEstimator,
    score_n_estimators: int,
    split: str = "random",
    seed: int = 42,
    test_size: float = 0.2,
    force: bool = False,
    fp_size: int = DEFAULT_FP_SIZE,
    fingerprints: list[str] | None = None,
    yscramble: bool = False,
) -> SklearnFingerprintTrainResult:
    """Train a sklearn fingerprint classifier (fixed recipe, no TPE)."""
    parent = Path(outdir)
    outdir = resolve_run_dir(
        parent,
        architecture,
        split,
        seed=seed,
        yscramble=yscramble,
    )
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
        architecture=architecture,
        cleaned_hash=prepared.cleaned_hash,
        yscramble=True if yscramble else None,
    )

    if ckpt.run_complete(prepared.stored, expected):
        logger.info(f"Reusing completed run in {ckpt.outdir}")
        classifier = classifier_cls.load(ckpt.outdir)
        report = TrainingReport.model_validate(ckpt.load_json(ckpt.report_json_path))
        return SklearnFingerprintTrainResult(
            model=classifier.model,
            classifier=classifier,
            report=report,
            outdir=ckpt.outdir,
        )

    cleaned = prepared.cleaned
    train_idx = prepared.train_idx
    test_idx = prepared.test_idx
    generators = {name: available[name] for name in selected}
    y_true = cleaned[CleanedTrainingData.LABEL]
    y = (
        scramble_train_labels(y_true, train_idx, seed)
        if yscramble
        else y_true
    )
    if yscramble:
        logger.info("Y-scramble: permuted train labels; test labels unchanged")
    fingerprint_comparison: dict[str, FingerprintScore] = {}
    stored = prepared.stored
    dumped = to_jsonable(params.model_dump())

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
        logger.info(f"Validating {fp_name} on held-out test split (no CV)")
        model, test_metrics = _fit_fingerprint_model(
            features,
            y,
            train_idx,
            test_idx,
            params=params,
            make_estimator=make_estimator,
            features_for_model=classifier_cls.features_for_model,
            y_test=y_true,
        )
        logger.info(f"{fp_name}: test ROC-AUC={test_metrics.roc_auc:.4f}")
        fingerprint_comparison[fp_name] = FingerprintScore(
            cv_auc=None,
            n_estimators=score_n_estimators,
            params=dumped,
            test_metrics=test_metrics,
        )
        metadata = ModelMeta(
            architecture=architecture,
            fingerprint=fp_name,
            fp_size=fp_size,
            params=dumped,
            n_estimators=score_n_estimators,
        )
        ckpt.fingerprint_dir(fp_name).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ckpt.fingerprint_sklearn_path(fp_name))
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
        architecture=architecture,
        yscramble=yscramble,
    )
    write_report(report, ckpt.report_json_path, ckpt.report_md_path)
    ckpt.save_config(expected)

    classifier = classifier_cls.load(ckpt.outdir)
    return SklearnFingerprintTrainResult(
        model=classifier.model,
        classifier=classifier,
        report=report,
        outdir=ckpt.outdir,
    )
