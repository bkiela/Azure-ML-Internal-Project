import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import umap

model_dir = r'azure-ml-internal-project\pipeline\training\models'
umap_model_path = os.path.join(model_dir, 'umap_model.pkl')

with open(umap_model_path, 'rb') as umap_file:
    loaded_umap = pickle.load(umap_file)

def create_scatterplots(input_dir, output_dir):
    scatterplot_save_dir = os.path.join(output_dir)

    for week_number in range(25, 31):
        week = f'week_{week_number}_year_2022.csv'
        input_csv = os.path.join(input_dir, week)
        output_csv = f'encoded_data_{week}'
        
        data = pd.read_csv(input_csv)
        country_region = data['Country_Region']
        selected_features = data.drop(['Country_Region'], axis=1)
        
        scaler = StandardScaler()
        selected_features_scaled = scaler.fit_transform(selected_features)
        
        new_selected_features_scaled = loaded_umap.transform(selected_features_scaled)
        new_umap_df = pd.DataFrame(new_selected_features_scaled, columns=['x', 'y'])
        new_umap_df['Country_Region'] = data['Country_Region']
        
        plt.figure(figsize=(16, 8))
        plt.scatter(new_umap_df['x'], new_umap_df['y'], c='blue', alpha=0.7)
        
        for i, row in new_umap_df.iterrows():
            plt.text(row['x'], row['y'], row['Country_Region'], fontsize=8)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Scatter Plot - {week}')
        
        scatterplot_save_path = os.path.join(scatterplot_save_dir, f'scatterplot_{week}_UMAP.png')
        plt.savefig(scatterplot_save_path)
        plt.show()

input_dir = r'azure-ml-internal-project\data\aggregated'
output_dir = r'azure-ml-internal-project\pipeline\evaluation\plots'

if __name__ == '__main__':
    create_scatterplots(input_dir, output_dir)
