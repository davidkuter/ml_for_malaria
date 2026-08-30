from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from ml_for_malaria.chemistry.featurization import DEFAULT_FP_SIZE
from ml_for_malaria.model.knn_classifier import ARCHITECTURE, KNNFingerprintClassifier
from ml_for_malaria.schemas import KNNParams
from ml_for_malaria.train.sklearn_fingerprint import (
    SklearnFingerprintTrainResult,
    train_sklearn_fingerprint_classifier,
)

DEFAULT_N_NEIGHBORS = 5


def default_knn_params(n_neighbors: int = DEFAULT_N_NEIGHBORS) -> KNNParams:
    """Fixed Tanimoto k-NN recipe (binary Jaccard distance, distance weights)."""
    return KNNParams(n_neighbors=int(n_neighbors))


def _make_knn(params: KNNParams, n_train: int) -> KNeighborsClassifier:
    n_neighbors = min(params.n_neighbors, n_train)
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=params.metric,
        weights=params.weights,
        algorithm=params.algorithm,
        n_jobs=params.n_jobs,
    )


def train_knn_classifier(
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
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
) -> SklearnFingerprintTrainResult:
    """Train a Tanimoto k-NN fingerprint classifier.

    Neighbours use Jaccard distance on binary fingerprints (1 − Tanimoto).
    ``max_evals>=1`` is rejected: this baseline is a fixed recipe.
    ``yscramble=True`` permutes train labels only; test labels stay real.
    """
    if max_evals > 0:
        raise ValueError("Tanimoto k-NN uses a fixed recipe; do not combine with HPO")
    params = default_knn_params(n_neighbors)
    return train_sklearn_fingerprint_classifier(
        df,
        outdir,
        architecture=ARCHITECTURE,
        classifier_cls=KNNFingerprintClassifier,
        params=params,
        make_estimator=_make_knn,
        score_n_estimators=params.n_neighbors,
        split=split,
        seed=seed,
        test_size=test_size,
        force=force,
        fp_size=fp_size,
        fingerprints=fingerprints,
        yscramble=yscramble,
    )
