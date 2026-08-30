from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_for_malaria.chemistry.featurization import DEFAULT_FP_SIZE
from ml_for_malaria.model.logistic_classifier import (
    ARCHITECTURE,
    LogisticFingerprintClassifier,
)
from ml_for_malaria.schemas import LogisticParams
from ml_for_malaria.train.sklearn_fingerprint import (
    SklearnFingerprintTrainResult,
    train_sklearn_fingerprint_classifier,
)


def default_logistic_params(seed: int) -> LogisticParams:
    """Fixed L2-logistic recipe used as a linear fingerprint baseline."""
    return LogisticParams(random_state=int(seed))


def _make_logistic(params: LogisticParams, _n_train: int) -> LogisticRegression:
    return LogisticRegression(**params.model_dump())


def train_logistic_classifier(
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
) -> SklearnFingerprintTrainResult:
    """Train an L2-logistic fingerprint classifier.

    ``max_evals>=1`` is rejected: this baseline is a fixed recipe.
    ``yscramble=True`` permutes train labels only; test labels stay real.
    """
    if max_evals > 0:
        raise ValueError("L2-logistic uses a fixed recipe; do not combine with HPO")
    params = default_logistic_params(seed)
    return train_sklearn_fingerprint_classifier(
        df,
        outdir,
        architecture=ARCHITECTURE,
        classifier_cls=LogisticFingerprintClassifier,
        params=params,
        make_estimator=_make_logistic,
        score_n_estimators=1,
        split=split,
        seed=seed,
        test_size=test_size,
        force=force,
        fp_size=fp_size,
        fingerprints=fingerprints,
        yscramble=yscramble,
    )
