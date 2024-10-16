import pandas as pd
import os
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import pickle

input_dir = r'azure-ml-internal-project\data\aggregated'
output_dir = r'azure-ml-internal-project\pipeline\training\training'
model_dir = r'azure-ml-internal-project\pipeline\training\models'

csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]
dfs = []

for filename in csv_files:
    filepath = os.path.join(input_dir, filename)
    temp_df = pd.read_csv(filepath)
    dfs.append(temp_df)
    
dfs = pd.concat(dfs, ignore_index=True)
country_region = dfs['Country_Region']
selected_features = dfs.drop(['Country_Region'], axis=1)

scaler = StandardScaler()
selected_features_scaled = scaler.fit_transform(selected_features)

scaler_filename = os.path.join(model_dir, 'ica_scaler.pkl')

with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
    
ica = FastICA(n_components=2)
ica_results = ica.fit_transform(selected_features_scaled, country_region)

ica_filename = os.path.join(model_dir, 'ica_model.pkl')

with open(ica_filename, 'wb') as ica_file:
    pickle.dump(ica, ica_file)
    
    