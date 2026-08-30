from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
    HyperoptInjected,
    HyperoptObjectiveResult,
    ModelMeta,
    RandomForestParams,
    RFHyperoptResult,
    RFMaxFeatures,
    RFSearchParam,
    RunConfig,
    SklearnClassWeight,
    TrainingReport,
)
from ml_for_malaria.train.fingerprint_features import fingerprint_features_for_run
from ml_for_malaria.train.prepare import prepare_training_run, scramble_train_labels

DEFAULT_N_ESTIMATORS = 200
HYPEROPT_EVALS = 100
_RF_CV_FOLDS = 5
_INT_RF_PARAMS = (
    RandomForestParams.n_estimators,
    RandomForestParams.min_samples_leaf,
)


@dataclass
class RFTrainResult:
    model: RandomForestClassifier
    classifier: RFFingerprintClassifier
    report: TrainingReport
    outdir: Path


def default_rf_params(seed: int, n_estimators: int = DEFAULT_N_ESTIMATORS) -> RandomForestParams:
    """Fixed forest recipe used when ``max_evals=0`` (no TPE / k-fold CV)."""
    return RandomForestParams(
        n_estimators=int(n_estimators),
        random_state=int(seed),
    )


def _int_rf_params(params: dict) -> dict:
    converted = dict(params)
    for key in _INT_RF_PARAMS:
        if key in converted and converted[key] is not None:
            converted[key] = int(converted[key])
    depth = converted.get(RandomForestParams.max_depth)
    if depth is not None:
        converted[RandomForestParams.max_depth] = int(depth) or None
    return converted


def _rf_params_from_search(best: dict, seed: int) -> RandomForestParams:
    converted = _int_rf_params(best)
    max_features = (
        RFMaxFeatures.LOG2
        if int(converted.pop(RFSearchParam.MAX_FEATURES_IX, 0)) == 1
        else RFMaxFeatures.SQRT
    )
    class_weight = (
        SklearnClassWeight.BALANCED
        if int(converted.pop(RFSearchParam.BALANCED_IX, 1)) == 1
        else None
    )
    return RandomForestParams(
        n_estimators=int(converted[RandomForestParams.n_estimators]),
        max_depth=converted.get(RandomForestParams.max_depth),
        min_samples_leaf=int(converted[RandomForestParams.min_samples_leaf]),
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=1,
        random_state=int(seed),
    )


def rf_hyperparameterisation(params: dict) -> dict:
    """Hyperopt objective: 5-fold stratified ROC-AUC on train rows only."""
    params = _int_rf_params(dict(params))
    features = params.pop(HyperoptInjected.features)
    labels = params.pop(HyperoptInjected.labels)
    seed = int(params.pop(RandomForestParams.random_state))
    search = _rf_params_from_search(params, seed)
    clf = RandomForestClassifier(**search.model_dump())
    folds = StratifiedKFold(n_splits=_RF_CV_FOLDS, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf,
        features,
        labels,
        cv=folds,
        scoring="roc_auc",
    )
    auc = float(np.nanmean(scores)) if len(scores) else 0.0
    if np.isnan(auc):
        auc = 0.0
    return HyperoptObjectiveResult(
        loss=-auc,
        status=STATUS_OK,
        n_estimators=search.n_estimators,
        auc=auc,
    ).model_dump()


def train_rf_cross_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    seed: int,
    max_evals: int = HYPEROPT_EVALS,
) -> RFHyperoptResult:
    """TPE search on train folds; returns forest params and CV AUC."""
    logger.info("Hyperparameter optimisation")
    space = {
        HyperoptInjected.features: features,
        HyperoptInjected.labels: labels,
        RandomForestParams.random_state: seed,
        RandomForestParams.n_estimators: hp.quniform(
            RandomForestParams.n_estimators, 50, 400, 10
        ),
        RandomForestParams.max_depth: hp.quniform(
            RandomForestParams.max_depth, 0, 20, 1
        ),
        RandomForestParams.min_samples_leaf: hp.quniform(
            RandomForestParams.min_samples_leaf, 1, 8, 1
        ),
        RFSearchParam.MAX_FEATURES_IX: hp.quniform(
            RFSearchParam.MAX_FEATURES_IX, 0, 1, 1
        ),
        RFSearchParam.BALANCED_IX: hp.quniform(RFSearchParam.BALANCED_IX, 0, 1, 1),
    }
    trials = Trials()
    best = fmin(
        fn=rf_hyperparameterisation,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )
    trial_result = HyperoptObjectiveResult.model_validate(trials.best_trial["result"])
    params = _rf_params_from_search(best, seed)
    return RFHyperoptResult(
        params=params,
        cv_auc=trial_result.auc,
        n_estimators=params.n_estimators,
    )


def _fit_fingerprint_model(
    features: pd.DataFrame,
    y: pd.Series,
    train_idx: list[int],
    test_idx: list[int],
    params: RandomForestParams,
    *,
    y_test: pd.Series | None = None,
) -> tuple[RandomForestClassifier, EvalMetrics]:
    clf = RandomForestClassifier(**params.model_dump())
    clf.fit(features.iloc[train_idx], y.iloc[train_idx])
    y_eval = y if y_test is None else y_test
    y_pred = clf.predict(features.iloc[test_idx])
    y_proba = clf.predict_proba(features.iloc[test_idx])[:, 1]
    test_metrics = compute_test_metrics(y_eval.iloc[test_idx], y_pred, y_proba)
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
    max_evals: int = 0,
    yscramble: bool = False,
) -> RFTrainResult:
    """Train a random-forest fingerprint classifier with checkpointed artifacts.

    ``df`` must contain SMILES and LABEL (0/1) columns.
    ``outdir`` is the parent runs directory; artifacts go in
    ``{outdir}/{arch}_{split}[_hpo][_yscramble]/seed_{seed}/``. ``max_evals=0`` uses a fixed
    forest. ``max_evals>=1`` runs TPE on train-fold ROC-AUC and selects the
    default artifact by CV AUC. Held-out test metrics are never the selector.
    ``yscramble=True`` permutes train labels only; test labels stay real.
    """
    if yscramble and max_evals > 0:
        raise ValueError("Y-scramble uses the fixed recipe; do not combine with HPO")
    parent = Path(outdir)
    skip_hyperopt = max_evals <= 0
    outdir = resolve_run_dir(
        parent,
        ARCHITECTURE,
        split,
        seed=seed,
        hpo=not skip_hyperopt,
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
        architecture=ARCHITECTURE,
        cleaned_hash=prepared.cleaned_hash,
        max_evals=max_evals if max_evals > 0 else None,
        yscramble=True if yscramble else None,
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
        if skip_hyperopt:
            params = default_rf_params(seed)
            logger.info(f"Validating {fp_name} on held-out test split (no CV)")
            forest, test_metrics = _fit_fingerprint_model(
                features, y, train_idx, test_idx, params, y_test=y_true
            )
            logger.info(f"{fp_name}: test ROC-AUC={test_metrics.roc_auc:.4f}")
            fingerprint_comparison[fp_name] = FingerprintScore(
                cv_auc=None,
                n_estimators=params.n_estimators,
                params=params.model_dump(),
                test_metrics=test_metrics,
            )
        else:
            hp_path = ckpt.hyperopt_path(fp_name)
            if ckpt.should_reuse(
                hp_path,
                stored,
                {
                    RunConfig.cleaned_hash: prepared.cleaned_hash,
                    RunConfig.split: split,
                    RunConfig.seed: seed,
                    RunConfig.test_size: test_size,
                    RunConfig.fp_size: fp_size,
                    RunConfig.max_evals: max_evals,
                },
            ):
                logger.info(f"Loading hyperopt results from {hp_path}")
                hp_result = RFHyperoptResult.model_validate(ckpt.load_json(hp_path))
            else:
                hp_result = train_rf_cross_validation(
                    features.iloc[train_idx],
                    y.iloc[train_idx],
                    seed=seed,
                    max_evals=max_evals,
                )
                ckpt.save_json(hp_path, hp_result)
            params = hp_result.params
            logger.info(f"Validating {fp_name} on held-out test split")
            forest, test_metrics = _fit_fingerprint_model(
                features, y, train_idx, test_idx, params, y_test=y_true
            )
            logger.info(
                f"{fp_name}: CV AUC={hp_result.cv_auc:.4f} "
                f"test ROC-AUC={test_metrics.roc_auc:.4f}"
            )
            fingerprint_comparison[fp_name] = FingerprintScore(
                cv_auc=hp_result.cv_auc,
                n_estimators=params.n_estimators,
                params=params.model_dump(),
                test_metrics=test_metrics,
            )
        metadata = ModelMeta(
            architecture=ARCHITECTURE,
            fingerprint=fp_name,
            fp_size=fp_size,
            params=to_jsonable(params.model_dump()),
            n_estimators=int(fingerprint_comparison[fp_name].n_estimators),
        )
        ckpt.fingerprint_dir(fp_name).mkdir(parents=True, exist_ok=True)
        joblib.dump(forest, ckpt.fingerprint_sklearn_path(fp_name))
        ckpt.save_json(ckpt.fingerprint_meta_path(fp_name), metadata)

    if skip_hyperopt:
        best_fingerprint = default_saved_fingerprint(selected)
        logger.info(f"Default fingerprint artifact: {best_fingerprint}")
    else:
        comparison = pd.DataFrame(
            {
                name: {FingerprintScore.cv_auc: score.cv_auc}
                for name, score in fingerprint_comparison.items()
            }
        ).T
        best_fingerprint = comparison[FingerprintScore.cv_auc].idxmax()
        best_cv = fingerprint_comparison[best_fingerprint].cv_auc
        logger.info(f"Best fingerprint by CV AUC: {best_fingerprint} ({best_cv:.4f})")
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
        max_evals=max_evals if max_evals > 0 else None,
        yscramble=yscramble,
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
