import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pickle
import random

source_dir = r'azure-ml-internal-project\data\aggregated'
destin_dir = r'azure-ml-internal-project\pipeline\training\training'
model_dir = r'azure-ml-internal-project\pipeline\training\models'

csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

dfs = []

for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    date_str = os.path.splitext(filename)[0]
    
    temp_df = pd.read_csv(filepath)
        
    dfs.append(temp_df)
dfs = pd.concat(dfs, ignore_index=True)
country_region = dfs['Country_Region']
year = dfs['Year']
selected_features = dfs.drop(['Country_Region', 'Year'], axis=1)

scaler = StandardScaler()
selected_features_scaled = scaler.fit_transform(selected_features)

scaler_filename = os.path.join(model_dir, 'pca_scaler.pkl')
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

pca = PCA(n_components=2)
pca.fit(selected_features_scaled)

random_csv_filename = random.choice(csv_files)
random_csv_path = os.path.join(source_dir, random_csv_filename)

random_csv = pd.read_csv(random_csv_path)
new_selected_features = random_csv.drop(['Country_Region', 'Year'], axis=1)
new_selected_features_scaled = scaler.transform(new_selected_features)
new_pca_results = pca.transform(new_selected_features_scaled)

new_pca_df = pd.DataFrame(new_pca_results, columns=['x', 'y'])
new_pca_df['Country_Region'] = random_csv['Country_Region']

output_filename = os.path.splitext(random_csv_filename)[0] + "_PCA.csv"
output_path = os.path.join(destin_dir, output_filename)
new_pca_df.to_csv(output_path, index=False)
