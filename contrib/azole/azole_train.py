import pandas as pd
from loguru import logger

from ml_for_malaria.train.train_xgb_classifier import train_classifier

DATASET_PATH = "../azole_prediction/data/100nM_Training_Set.csv"
MODEL_PATH = "./azole_model"

# Load data
logger.info(f"Loading data from: {DATASET_PATH}")
df_input = pd.read_csv(DATASET_PATH)
df_input = df_input[["SMILES", "Lable"]]
df_input = df_input.rename(columns={"Lable": "LABEL"})
df_input["LABEL"] = df_input["LABEL"].apply(lambda x: 1 if x == "Active" else 0)

# Train model
model = train_classifier(df=df_input, model_outpath=MODEL_PATH, save_features=False)

# Past best model stats
# - We removed this because Topological Fingerprints are not interpretable
# Topological AUC: 0.914665035423063
# Params: {'alpha': 11, 'colsample_bytree': 0.9819434248719625, 'gamma': 1.3009905530433081,
#          'learning_rate': 0.06722296035314712, 'max_depth': 9, 'min_child_weight': 0,
#          'reg_lambda': 0.5015487969249303}
# Model accuracy score: 0.868

