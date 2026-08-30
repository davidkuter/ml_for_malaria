from pathlib import Path

import pandas as pd
from loguru import logger

from ml_for_malaria.chemistry import encode_binary_labels
from ml_for_malaria.model.smiles_transformer import DEFAULT_PRETRAINED_NAME
from ml_for_malaria.schemas import CleanedTrainingData
from ml_for_malaria.train.chemberta import train_smiles_transformer

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

result = train_smiles_transformer(
    df=df_input,
    outdir=RUNS,
    split="scaffold",
    seed=42,
    pretrained_name=DEFAULT_PRETRAINED_NAME,
    freeze_encoder=True,
    force=False,
)
logger.info(f"Wrote report to {result.outdir / 'report.md'}")
