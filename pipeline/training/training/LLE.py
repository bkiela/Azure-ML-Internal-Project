import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding
import os
import pickle

source_dir = r'azure-ml-internal-project\data\aggregated'
destin_dir = r'azure-ml-internal-project\pipeline\training\training'
model_dir = r'azure-ml-internal-project\pipeline\training\models'

csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

dfs = []

for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    temp_df = pd.read_csv(filepath)
    dfs.append(temp_df)
dfs = pd.concat(dfs, ignore_index=True)
country_region = dfs['Country_Region']
selected_features = dfs.drop(['Country_Region'], axis=1)

scaler = StandardScaler()
selected_features_scaled = scaler.fit_transform(selected_features)

scaler_filename = os.path.join(model_dir, 'lle_scaler.pkl')
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

lle = LocallyLinearEmbedding(n_components=2, eigen_solver='dense')
lle_results = lle.fit_transform(selected_features_scaled)

lle_filename = os.path.join(model_dir, 'lle_model.pkl')
with open(lle_filename, 'wb') as lle_file:
    pickle.dump(lle, lle_file)
