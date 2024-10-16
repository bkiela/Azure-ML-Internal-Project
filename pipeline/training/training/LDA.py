import os
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

scaler_filename = os.path.join(model_dir, 'lda_scaler.pkl')
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
    
lda = LinearDiscriminantAnalysis(n_components=2)
lda_results = lda.fit_transform(selected_features_scaled, country_region)

lda_filename = os.path.join(model_dir, 'lda_model.pkl')
with open(lda_filename, 'wb') as lda_file:
    pickle.dump(lda, lda_file)

