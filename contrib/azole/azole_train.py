import pandas as pd
from loguru import logger
from pathlib import Path

from model.azole import AzoleModel
from train.train_classifier import train_classifier, prepare_data, feature_importance
from train.featurisation import sanitise_smiles


DATASET_PATH = './data/100nM_Training_Set.csv'
MODEL_PATH = './azole_model'

# Load data
logger.info(f"Loading data from: {DATASET_PATH}")
df_input = pd.read_csv(DATASET_PATH)
df_input = df_input[['SMILES', 'Lable']]
df_input = df_input.rename(columns={'Lable': 'LABEL'})
df_input['LABEL'] = df_input['LABEL'].apply(lambda x: 1 if x == 'Active' else 0)

# Train model
# model = train_classifier(df=df_input, model_outpath=MODEL_PATH, save_features=False)

# Past best model stats


# - We removed this because Topological Fingerprints are not interpretable
# Topological AUC: 0.914665035423063
# Params: {'alpha': 11, 'colsample_bytree': 0.9819434248719625, 'gamma': 1.3009905530433081,
#          'learning_rate': 0.06722296035314712, 'max_depth': 9, 'min_child_weight': 0,
#          'reg_lambda': 0.5015487969249303}
# Model accuracy score: 0.868

######### Testing delete afterwards
model = AzoleModel()
model.load_model(model_path=f"{MODEL_PATH}.ubj")

# Sanitize SMILES
df = df_input.rename(columns={'SMILES': 'INPUT_SMILES'})
df['SMILES'] = df['INPUT_SMILES'].apply(lambda x: sanitise_smiles(x, as_mol=False))
df = df.dropna(subset=['SMILES'])
df = df.drop_duplicates(subset=['SMILES'])

# _, _, X, y, _, _, _, _ = prepare_data(df=df, generator=model.feature_generator, seed=42)
# feature_importance(xgb_clf=model.model, X=X, out_dir=Path('./'))
inactive_smiles = 'O=C(c1conc1)N1CC[C@H](C1)Nc1nccc(n1)c1c(C)onc1c1ccc(Cl)cc1'
model.get_feature_importance(smiles=inactive_smiles, img_out='./simmap_inactive.png')
active_smiles = 'CS(=O)(=O)c1cccc(c1)c1nc2cc(ccn2c1c1ccnc(NCC2CC2)n1)CN(C)C'
model.get_feature_importance(smiles=active_smiles, img_out='./simmap_active.png')
