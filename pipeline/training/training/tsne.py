import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import pickle

source_dir = r'azure-ml-internal-project\data\aggregated'
destin_dir = r'azure-ml-internal-project\pipeline\training\training'
model_dir = r'azure-ml-internal-project\pipeline\training\models'

class TSNE_abstract():
    model = None
    
    def __init__(self,n_components=2):
        self.model = TSNE(n_components=n_components)
    
    def fit_transform(self,selected_features_scaled):
        return self.model.fit_transform(selected_features_scaled)
    
    def transform(self,selected_features_scaled):
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

scaler_filename = os.path.join(model_dir, 'tsne_scaler.pkl')
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

tsne = TSNE_abstract(n_components=2)
tsne_results = tsne.fit_transform(selected_features_scaled)

tsne_filename = os.path.join(model_dir, 'tsne_model.pkl')
with open(tsne_filename, 'wb') as tsne_file:
    pickle.dump(tsne, tsne_file)

