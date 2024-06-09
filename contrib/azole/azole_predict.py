import pandas as pd

from model.azole import AzoleModel

data_path = './data/100nM_Experimental_Azoles.csv'
model_path = 'azole_model.ubj'
out_path = './azole_results.csv'

df = pd.read_csv(data_path)
df = df.rename(columns={'Smiles': 'SMILES'})

model = AzoleModel()
model.load_model(model_path=model_path)
df_results = model.predict(smiles=df['SMILES'].to_list())
df = df.merge(df_results, on='SMILES', how='left')
df = df.sort_values(by=['PROBABILITY'], ascending=False)
df.to_csv(out_path, index=False)
print(df)
