from pathlib import Path

import pandas as pd
from loguru import logger

from ml_for_malaria.schemas import CleanedTrainingData
from ml_for_malaria.train.featurization import encode_binary_labels
from ml_for_malaria.train.train_chemprop import train_chemprop_classifier

ROOT = Path(__file__).resolve().parents[2]
PFPKG = ROOT / "data" / "pfpkg"
DATASET_PATH = PFPKG / "input" / "100nM_Training_Set.csv"
RUNS = PFPKG / "runs" / "pfpkg"

logger.info(f"Loading data from: {DATASET_PATH}")
df_input = pd.read_csv(DATASET_PATH)
df_input = df_input.rename(columns={"Lable": CleanedTrainingData.LABEL})
df_input = df_input[[CleanedTrainingData.SMILES, CleanedTrainingData.LABEL]]
df_input[CleanedTrainingData.LABEL] = encode_binary_labels(
    df_input[CleanedTrainingData.LABEL],
    active_label="Active",
    inactive_label="Inactive",
)

result = train_chemprop_classifier(
    df=df_input,
    outdir=RUNS,
    split="scaffold",
    seed=42,
    charge_method=None,
    force=False,
)
logger.info(f"Wrote report to {result.outdir / 'report.md'}")
