from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from ml_for_malaria.chemistry.featurization import clean_training_data
from ml_for_malaria.runs.checkpoints import RunCheckpointer, data_hash
from ml_for_malaria.schemas import CleanedTrainingData, RunConfig, SplitIndices
from ml_for_malaria.split import get_splitter

HASH_COLUMNS = [CleanedTrainingData.SMILES, CleanedTrainingData.LABEL]


@dataclass
class PreparedRun:
    ckpt: RunCheckpointer
    cleaned: pd.DataFrame
    train_idx: list[int]
    test_idx: list[int]
    stored: RunConfig | None
    input_hash: str
    cleaned_hash: str


def prepare_training_run(
    df: pd.DataFrame,
    outdir: str | Path,
    *,
    split: str = "random",
    seed: int = 42,
    test_size: float = 0.2,
    force: bool = False,
    ckpt: RunCheckpointer | None = None,
) -> PreparedRun:
    """Sanitize labels/SMILES and freeze a train/test split under ``outdir``."""
    ckpt = ckpt or RunCheckpointer(outdir, force=force)
    stored = ckpt.load_config()
    input_hash = data_hash(df, HASH_COLUMNS)
    splitter = get_splitter(split)

    if ckpt.should_reuse(ckpt.cleaned_path, stored, {RunConfig.input_hash: input_hash}):
        cleaned = ckpt.load_cleaned()
    else:
        logger.info("Cleaning training data")
        cleaned = clean_training_data(df)
        ckpt.save_cleaned(cleaned)

    cleaned_hash = data_hash(cleaned, HASH_COLUMNS)
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
        ckpt.save_json(split_file, SplitIndices(train_idx=train_idx, test_idx=test_idx))

    return PreparedRun(
        ckpt=ckpt,
        cleaned=cleaned,
        train_idx=train_idx,
        test_idx=test_idx,
        stored=stored,
        input_hash=input_hash,
        cleaned_hash=cleaned_hash,
    )


def train_val_indices(
    train_idx: list[int],
    labels: pd.Series,
    seed: int,
    val_size: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Carve a validation slice from train indices only (never from test)."""
    positions = np.asarray(train_idx)
    if len(positions) < 4:
        return positions.tolist(), positions[: max(1, len(positions) // 2)].tolist()
    y = labels.iloc[positions]
    try:
        fit_idx, val_idx = train_test_split(
            positions,
            test_size=val_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        fit_idx, val_idx = train_test_split(
            positions,
            test_size=val_size,
            random_state=seed,
        )
    return fit_idx.tolist(), val_idx.tolist()
