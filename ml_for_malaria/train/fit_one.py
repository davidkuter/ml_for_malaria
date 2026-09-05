from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from ml_for_malaria.runs.checkpoints import RunCheckpointer
from ml_for_malaria.schemas import Architecture
from ml_for_malaria.train import (
    train_knn_classifier,
    train_logistic_classifier,
    train_rf_classifier,
    train_xgb_classifier,
)


def fit_one_run(
    seed: int,
    *,
    architecture: str,
    df: pd.DataFrame,
    outdir: Path,
    split: str,
    charge_method: str | None = None,
    fingerprints: list[str] | None = None,
    max_evals: int = 0,
    yscramble: bool = False,
    force: bool = False,
) -> Path | None:
    """Fit one architecture/seed. Returns the run dir, or None if a DL extra is missing."""
    try:
        if architecture == Architecture.CHEMBERTA:
            from ml_for_malaria.model.smiles_transformer import DEFAULT_PRETRAINED_NAME
            from ml_for_malaria.train.chemberta import train_smiles_transformer

            result = train_smiles_transformer(
                df=df,
                outdir=outdir,
                split=split,
                seed=seed,
                pretrained_name=DEFAULT_PRETRAINED_NAME,
                freeze_encoder=True,
                force=force,
                yscramble=yscramble,
            )
        elif architecture == Architecture.CHEMPROP:
            from ml_for_malaria.train.chemprop import train_chemprop_classifier

            result = train_chemprop_classifier(
                df=df,
                outdir=outdir,
                split=split,
                seed=seed,
                charge_method=charge_method,
                force=force,
                yscramble=yscramble,
            )
        elif architecture == Architecture.CHEMELEON:
            from ml_for_malaria.schemas import FoundationModel
            from ml_for_malaria.train.chemprop import train_chemprop_classifier

            result = train_chemprop_classifier(
                df=df,
                outdir=outdir,
                split=split,
                seed=seed,
                foundation=FoundationModel.CHEMELEON,
                force=force,
                yscramble=yscramble,
            )
        elif architecture == Architecture.MONROE:
            from ml_for_malaria.train.monroe import train_monroe_classifier

            result = train_monroe_classifier(
                df=df,
                outdir=outdir,
                split=split,
                seed=seed,
                force=force,
                yscramble=yscramble,
            )
        elif architecture == Architecture.RANDOM_FOREST:
            result = train_rf_classifier(
                df=df,
                outdir=outdir,
                split=split,
                seed=seed,
                force=force,
                fingerprints=fingerprints,
                max_evals=max_evals,
                yscramble=yscramble,
            )
        elif architecture == Architecture.XGBOOST:
            result = train_xgb_classifier(
                df=df,
                outdir=outdir,
                split=split,
                seed=seed,
                force=force,
                fingerprints=fingerprints,
                max_evals=max_evals,
                yscramble=yscramble,
            )
        elif architecture == Architecture.KNN:
            result = train_knn_classifier(
                df=df,
                outdir=outdir,
                split=split,
                seed=seed,
                force=force,
                fingerprints=fingerprints,
                max_evals=max_evals,
                yscramble=yscramble,
            )
        elif architecture == Architecture.LOGISTIC:
            result = train_logistic_classifier(
                df=df,
                outdir=outdir,
                split=split,
                seed=seed,
                force=force,
                fingerprints=fingerprints,
                max_evals=max_evals,
                yscramble=yscramble,
            )
        else:
            raise ValueError(f"Unsupported suite architecture {architecture!r}")
    except ImportError as exc:
        logger.error(f"Skipping {architecture} seed={seed}: {exc}")
        return None
    logger.info(f"Ready {result.outdir / RunCheckpointer.REPORT_MD}")
    return result.outdir
