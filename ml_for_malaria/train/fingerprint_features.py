from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_for_malaria.chemistry.featurization import featurize_smiles
from ml_for_malaria.runs.checkpoints import RunCheckpointer
from ml_for_malaria.runs.shared_features import (
    load_shared_features,
    save_shared_features,
)
from ml_for_malaria.schemas import CleanedTrainingData, RunConfig


def fingerprint_features_for_run(
    *,
    ckpt: RunCheckpointer,
    parent: Path,
    fp_name: str,
    generator,
    cleaned: pd.DataFrame,
    cleaned_hash: str,
    fp_size: int,
    stored: RunConfig | None,
) -> pd.DataFrame:
    """Load run-local, then parent-shared fingerprints; compute and cache on miss."""
    feat_path = ckpt.features_path(fp_name)
    features = None
    if ckpt.should_reuse(
        feat_path,
        stored,
        {
            RunConfig.cleaned_hash: cleaned_hash,
            RunConfig.fp_size: fp_size,
        },
    ):
        features = ckpt.load_features(fp_name)
    if features is None:
        features = load_shared_features(parent, fp_name, fp_size, cleaned_hash)
    if features is None:
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
        save_shared_features(parent, fp_name, fp_size, cleaned_hash, features)
    features = features.reset_index(drop=True)
    ckpt.save_features(fp_name, features)
    return features
