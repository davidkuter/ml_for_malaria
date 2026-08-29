from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from loguru import logger
from xgboost import XGBClassifier

from ml_for_malaria.model.xgb_classifier import ARCHITECTURE, XGBFingerprintClassifier
from ml_for_malaria.schemas import (
    CleanedTrainingData,
    FingerprintScore,
    HyperoptInjected,
    HyperoptObjectiveResult,
    HyperoptResult,
    ModelMeta,
    RunConfig,
    SplitIndices,
    TrainingReport,
    XGBParams,
)
from ml_for_malaria.train.checkpoints import RunCheckpointer, data_hash, to_jsonable
from ml_for_malaria.train.featurization import (
    DEFAULT_FP_SIZE,
    clean_training_data,
    featurize_smiles,
    get_fingerprint_generator,
    get_fingerprint_generators,
)
from ml_for_malaria.train.report import build_report, compute_test_metrics, write_report
from ml_for_malaria.train.split import get_splitter

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 20
HYPEROPT_EVALS = 100
OBJECTIVE = "binary:logistic"
_XGB_CV_AUC = "test-auc-mean"
_HASH_COLUMNS = [CleanedTrainingData.SMILES, CleanedTrainingData.LABEL]
_INT_XGB_PARAMS = (
    XGBParams.alpha,
    XGBParams.max_depth,
    XGBParams.min_child_weight,
)


@dataclass
class XGBTrainResult:
    model: XGBClassifier
    classifier: XGBFingerprintClassifier
    report: TrainingReport
    outdir: Path


def _int_params(params: dict) -> dict:
    converted = dict(params)
    for key in _INT_XGB_PARAMS:
        if key in converted:
            converted[key] = int(converted[key])
    return converted


def hyperparameterisation(params: dict) -> dict:
    """Hyperopt objective: 5-fold XGBoost CV, loss = -AUC."""
    params = _int_params(dict(params))
    dtrain = params.pop(HyperoptInjected.dtrain)
    results = xgb.cv(
        dtrain=dtrain,
        params=params,
        nfold=5,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        metrics="auc",
        as_pandas=True,
        seed=int(params.get(XGBParams.seed, 0)),
    )
    auc_series = results[_XGB_CV_AUC]
    if auc_series.isna().all():
        return HyperoptObjectiveResult(
            loss=0.0,
            status=STATUS_OK,
            n_estimators=1,
            auc=0.0,
        ).model_dump()
    auc = float(auc_series.max())
    n_estimators = int(auc_series.idxmax()) + 1
    return HyperoptObjectiveResult(
        loss=-1 * auc,
        status=STATUS_OK,
        n_estimators=n_estimators,
        auc=auc,
    ).model_dump()


def _sklearn_params(best: dict, seed: int, n_estimators: int) -> XGBParams:
    return XGBParams(
        objective=OBJECTIVE,
        alpha=int(best[XGBParams.alpha]),
        gamma=float(best[XGBParams.gamma]),
        reg_lambda=float(best[XGBParams.reg_lambda]),
        colsample_bytree=float(best[XGBParams.colsample_bytree]),
        min_child_weight=int(best[XGBParams.min_child_weight]),
        max_depth=int(best[XGBParams.max_depth]),
        learning_rate=float(best[XGBParams.learning_rate]),
        n_estimators=int(n_estimators),
        random_state=int(seed),
        seed=int(seed),
        eval_metric="auc",
    )


def train_cross_validation_model(
    dtrain: xgb.DMatrix, seed: int, max_evals: int = HYPEROPT_EVALS
) -> HyperoptResult:
    """TPE hyperparameter search; returns sklearn params, CV AUC, and n_estimators."""
    logger.info("Hyperparameter optimisation")
    space = {
        XGBParams.objective: OBJECTIVE,
        XGBParams.eval_metric: "auc",
        HyperoptInjected.dtrain: dtrain,
        XGBParams.alpha: hp.quniform(XGBParams.alpha, 10, 200, 1),
        XGBParams.gamma: hp.uniform(XGBParams.gamma, 1, 9),
        XGBParams.reg_lambda: hp.uniform(XGBParams.reg_lambda, 0, 1),
        XGBParams.colsample_bytree: hp.uniform(XGBParams.colsample_bytree, 0.5, 1),
        XGBParams.min_child_weight: hp.quniform(XGBParams.min_child_weight, 0, 10, 1),
        XGBParams.max_depth: hp.uniformint(XGBParams.max_depth, 3, 18),
        XGBParams.learning_rate: hp.uniform(XGBParams.learning_rate, 0.01, 0.2),
        XGBParams.seed: seed,
    }
    trials = Trials()
    best = fmin(
        fn=hyperparameterisation,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )
    best = _int_params(best)
    trial_result = HyperoptObjectiveResult.model_validate(trials.best_trial["result"])
    params = _sklearn_params(best, seed, trial_result.n_estimators)
    return HyperoptResult(
        params=params,
        cv_auc=trial_result.auc,
        n_estimators=trial_result.n_estimators,
    )


def train_xgb_classifier(
    df: pd.DataFrame,
    outdir: str | Path,
    split: str = "random",
    seed: int = 42,
    test_size: float = 0.2,
    force: bool = False,
    fp_size: int = DEFAULT_FP_SIZE,
    max_evals: int = HYPEROPT_EVALS,
    fingerprints: list[str] | None = None,
) -> XGBTrainResult:
    """Train an XGBoost fingerprint classifier with checkpointed intermediate results.

    ``df`` must contain SMILES and LABEL (0/1) columns.
    """
    ckpt = RunCheckpointer(outdir, force=force)
    stored = ckpt.load_config()
    input_hash = data_hash(df, _HASH_COLUMNS)
    splitter = get_splitter(split)
    available = get_fingerprint_generators(fp_size=fp_size)
    selected = list(available) if fingerprints is None else list(fingerprints)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown fingerprint(s) {unknown}. Supported: {sorted(available)}"
        )

    expected = RunConfig(
        input_hash=input_hash,
        split=split,
        seed=seed,
        test_size=test_size,
        fp_size=fp_size,
        max_evals=max_evals,
        fingerprints=selected,
        architecture=ARCHITECTURE,
    )

    if ckpt.run_complete(stored, expected):
        logger.info(f"Reusing completed run in {ckpt.outdir}")
        classifier = XGBFingerprintClassifier.load(ckpt.outdir)
        report = TrainingReport.model_validate(ckpt.load_json(ckpt.report_json_path))
        return XGBTrainResult(
            model=classifier.model,
            classifier=classifier,
            report=report,
            outdir=ckpt.outdir,
        )

    if ckpt.should_reuse(
        ckpt.cleaned_path, stored, {RunConfig.input_hash: input_hash}
    ):
        cleaned = ckpt.load_cleaned()
    else:
        logger.info("Cleaning training data")
        cleaned = clean_training_data(df)
        ckpt.save_cleaned(cleaned)

    cleaned_hash = data_hash(cleaned, _HASH_COLUMNS)
    expected = expected.model_copy(
        update={RunConfig.cleaned_hash: cleaned_hash}
    )

    split_file = ckpt.split_path(split, seed)
    if ckpt.should_reuse(
        split_file,
        stored,
        {
            RunConfig.cleaned_hash: cleaned_hash,
            RunConfig.split: split,
            RunConfig.seed: seed,
            RunConfig.test_size: test_size,
        },
    ):
        split_payload = SplitIndices.model_validate(ckpt.load_json(split_file))
        train_idx = split_payload.train_idx
        test_idx = split_payload.test_idx
        logger.info(f"Loading split from {split_file}")
    else:
        logger.info(f"Splitting data with strategy={split!r}")
        train_idx, test_idx = splitter.split(
            smiles=cleaned[CleanedTrainingData.SMILES],
            labels=cleaned[CleanedTrainingData.LABEL],
            test_size=test_size,
            seed=seed,
        )
        ckpt.save_json(
            split_file, SplitIndices(train_idx=train_idx, test_idx=test_idx)
        )

    generators = {name: available[name] for name in selected}
    y = cleaned[CleanedTrainingData.LABEL]
    fingerprint_comparison: dict[str, FingerprintScore] = {}

    for fp_name, generator in generators.items():
        logger.info(f"Evaluating {fp_name}")
        feat_path = ckpt.features_path(fp_name)
        if ckpt.should_reuse(
            feat_path,
            stored,
            {
                RunConfig.cleaned_hash: cleaned_hash,
                RunConfig.fp_size: fp_size,
            },
        ):
            features = ckpt.load_features(fp_name)
        else:
            features = featurize_smiles(
                smiles=cleaned[CleanedTrainingData.SMILES],
                fp_generator=generator,
                sanitize=False,
            )
            if len(features) != len(cleaned):
                missing = len(cleaned) - len(features)
                raise RuntimeError(
                    f"{fp_name}: {missing} SMILES failed featurization after sanitization"
                )
            features = features.reset_index(drop=True)
            ckpt.save_features(fp_name, features)

        features = features.reset_index(drop=True)

        hp_path = ckpt.hyperopt_path(fp_name)
        if ckpt.should_reuse(
            hp_path,
            stored,
            {
                RunConfig.cleaned_hash: cleaned_hash,
                RunConfig.split: split,
                RunConfig.seed: seed,
                RunConfig.test_size: test_size,
                RunConfig.fp_size: fp_size,
                RunConfig.max_evals: max_evals,
            },
        ):
            logger.info(f"Loading hyperopt results from {hp_path}")
            hp_result = HyperoptResult.model_validate(ckpt.load_json(hp_path))
        else:
            dtrain = xgb.DMatrix(
                data=features.iloc[train_idx],
                label=y.iloc[train_idx],
            )
            hp_result = train_cross_validation_model(
                dtrain=dtrain, seed=seed, max_evals=max_evals
            )
            ckpt.save_json(hp_path, hp_result)

        fingerprint_comparison[fp_name] = FingerprintScore(
            cv_auc=hp_result.cv_auc,
            n_estimators=hp_result.n_estimators,
            params=hp_result.params.model_dump(),
        )

    comparison = pd.DataFrame.from_dict(
        {name: score.model_dump() for name, score in fingerprint_comparison.items()},
        orient="index",
    )
    best_fingerprint = comparison[FingerprintScore.cv_auc].idxmax()
    best = fingerprint_comparison[best_fingerprint]
    logger.info(
        f"Best fingerprint: {best_fingerprint}. CV AUC: {best.cv_auc:.4f}"
    )

    features = ckpt.load_features(best_fingerprint)
    params = XGBParams.model_validate(best.params).model_copy(
        update={XGBParams.n_estimators: int(best.n_estimators)}
    )
    x_train = features.iloc[train_idx]
    y_train = y.iloc[train_idx]
    x_test = features.iloc[test_idx]
    y_test = y.iloc[test_idx]

    logger.info("Validating model on held-out test split")
    xgb_clf = XGBClassifier(**params.model_dump())
    xgb_clf.fit(x_train, y_train)
    y_pred = xgb_clf.predict(x_test)
    y_proba = xgb_clf.predict_proba(x_test)[:, 1]
    test_metrics = compute_test_metrics(y_test, y_pred, y_proba)
    logger.info(
        f"Test accuracy={test_metrics.accuracy:.3f} "
        f"roc_auc={test_metrics.roc_auc:.3f} "
        f"weighted_f1={test_metrics.weighted.f1:.3f}"
    )

    report = build_report(
        split=split,
        seed=seed,
        test_size=test_size,
        n_train=len(train_idx),
        n_test=len(test_idx),
        best_fingerprint=best_fingerprint,
        fingerprint_comparison=fingerprint_comparison,
        test_metrics=test_metrics,
        architecture=ARCHITECTURE,
    )
    write_report(report, ckpt.report_json_path, ckpt.report_md_path)

    logger.info("Fitting final model on the full dataset")
    xgb_clf.fit(features, y)
    xgb_clf.save_model(str(ckpt.model_path))
    metadata = ModelMeta(
        architecture=ARCHITECTURE,
        fingerprint=best_fingerprint,
        fp_size=fp_size,
        params=to_jsonable(params.model_dump()),
        n_estimators=int(best.n_estimators),
    )
    ckpt.save_json(ckpt.meta_path, metadata)
    ckpt.save_config(expected)

    classifier = XGBFingerprintClassifier(
        model=xgb_clf,
        feature_generator=get_fingerprint_generator(best_fingerprint, fp_size),
        metadata=metadata,
    )
    return XGBTrainResult(
        model=xgb_clf,
        classifier=classifier,
        report=report,
        outdir=ckpt.outdir,
    )
