from pathlib import Path

import pandas as pd

from ml_for_malaria.model import XGBFingerprintClassifier

ROOT = Path(__file__).resolve().parents[2]
AZOLE = ROOT / "data" / "azole"
DATA_PATH = AZOLE / "input" / "100nM_Experimental_Azoles.csv"
OUTDIR = AZOLE / "runs"
RESULTS_PATH = OUTDIR / "azole_results.csv"

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"Smiles": "SMILES"})

model = XGBFingerprintClassifier.load(OUTDIR)
df_results = model.predict(df["SMILES"])
df = df.merge(df_results, on="SMILES", how="left")
df = df.sort_values(by=["PROBABILITY"], ascending=False)
df.to_csv(RESULTS_PATH, index=False)
print(df)
