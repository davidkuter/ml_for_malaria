from pathlib import Path

import pandas as pd

from ml_for_malaria.model import XGBFingerprintClassifier
from ml_for_malaria.schemas import Predictions

ROOT = Path(__file__).resolve().parents[2]
PFPKG = ROOT / "data" / "pfpkg"
DATA_PATH = PFPKG / "input" / "100nM_Experimental_Azoles.csv"
OUTDIR = PFPKG / "runs" / "xgb_random"
RESULTS_PATH = OUTDIR / "pfpkg_results.csv"

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"Smiles": Predictions.SMILES})

model = XGBFingerprintClassifier.load(OUTDIR)
df_results = model.predict(df[Predictions.SMILES])
df = df.merge(df_results, on=Predictions.SMILES, how="left")
df = df.sort_values(by=[Predictions.PROBABILITY], ascending=False)
df.to_csv(RESULTS_PATH, index=False)
print(df)
