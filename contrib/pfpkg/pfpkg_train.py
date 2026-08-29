from pathlib import Path

import pandas as pd
from loguru import logger

from ml_for_malaria.train import train_xgb_classifier
from ml_for_malaria.train.featurization import encode_binary_labels

ROOT = Path(__file__).resolve().parents[2]
AZOLE = ROOT / "data" / "azole"
DATASET_PATH = AZOLE / "input" / "100nM_Training_Set.csv"
OUTDIR = AZOLE / "runs"

logger.info(f"Loading data from: {DATASET_PATH}")
df_input = pd.read_csv(DATASET_PATH)
df_input = df_input.rename(columns={"Lable": "LABEL"})
df_input = df_input[["SMILES", "LABEL"]]
df_input["LABEL"] = encode_binary_labels(
    df_input["LABEL"], active_label="Active", inactive_label="Inactive"
)

result = train_xgb_classifier(
    df=df_input,
    outdir=OUTDIR,
    split="random",
    seed=42,
    force=False,
)
logger.info(f"Wrote report to {result.outdir / 'report.md'}")
logger.info(f"Best fingerprint: {result.report['best_fingerprint']}")
