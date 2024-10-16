import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

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

scaler_filename = os.path.join(model_dir, 'kpca_scaler.pkl')

with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
    
kpca = KernelPCA(n_components=2)
kpca_results = kpca.fit_transform(selected_features_scaled, country_region)

kpca_filename = os.path.join(model_dir, 'kpca_model.pkl')

with open(kpca_filename, 'wb') as kpca_file:
    pickle.dump(kpca, kpca_file)
    