import pandas as pd
import os
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import pickle

source_dir = r'azure-ml-internal-project\data\aggregated'
destin_dir = r'azure-ml-internal-project\pipeline\training\training'
model_dir = r'azure-ml-internal-project\pipeline\training\models'

class MDS_abstract():
    model = None
    
    def __init__(self, n_components=2):
        self.model = MDS(n_components=n_components)
        
    def fit_transform(self, selected_features_scaled):
        return self.model.fit_transform(selected_features_scaled)
    
    def transform(self, selected_features_scaled):
        return self.model.fit_transform(selected_features_scaled)

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

scaler_filename = os.path.join(model_dir, 'mds_scaler.pkl')

with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
    
mds = MDS_abstract(n_components=2)
mds_results = mds.fit_transform(selected_features_scaled)

mds_filename = os.path.join(model_dir, 'mds_model.pkl')
with open(mds_filename, 'wb') as mds_file:
    pickle.dump(mds, mds_file)
    
# Bardzo wolne, bardzo bardzo wolne
