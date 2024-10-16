import pandas as pd
import os
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
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

scaler_filename = os.path.join(model_dir, 'isomap_scaler.pkl')

with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

isomap = Isomap(n_components=2)
isomap_results = isomap.fit_transform(selected_features_scaled, country_region)

isomap_filename = os.path.join(model_dir, 'isomap_model.pkl')
with open(isomap_filename, 'wb') as isomap_file:
    pickle.dump(isomap, isomap_file)
    
    