from pathlib import Path

import pandas as pd

from ml_for_malaria.model import XGBFingerprintClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "azole" / "100nM_Experimental_Azoles.csv"
OUTDIR = ROOT / "runs" / "azole_100nM"
RESULTS_PATH = Path(__file__).resolve().parent / "azole_results.csv"

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"Smiles": "SMILES"})

model = XGBFingerprintClassifier.load(OUTDIR)
df_results = model.predict(smiles=df["SMILES"].to_list())
df = df.merge(df_results, on="SMILES", how="left")
df = df.sort_values(by=["PROBABILITY"], ascending=False)
df.to_csv(RESULTS_PATH, index=False)
print(df)
