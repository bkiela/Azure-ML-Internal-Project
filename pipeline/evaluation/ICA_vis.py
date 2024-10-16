import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

input_dir = r'azure-ml-internal-project\data\aggregated'
output_dir = r'azure-ml-internal-project\pipeline\evaluation\plots'
model_dir = r'azure-ml-internal-project\pipeline\training\models'
ica_model_path = os.path.join(model_dir, 'ica_model.pkl')

with open(ica_model_path, 'rb') as ica_file:
    loaded_ica = pickle.load(ica_file)
    
def create_scatterplots(input_dir, output_dir):
    scatterplot_save_dir = os.path.join(output_dir)
    
    for week_num in range(25, 31):
        week = f'week_{week_num}_year_2022.csv'
        input_csv = os.path.join(input_dir, week)
        output_dir = f'encoded_data_{week}_ica'
        
        data = pd.read_csv(input_csv)
        country_region = data['Country_Region']
        selected_features = data.drop(['Country_Region'], axis=1)
        
        scaler = StandardScaler()
        selected_features_scaled = scaler.fit_transform(selected_features)
        
        new_selected_features_scaled = loaded_ica.transform(selected_features_scaled)
        new_ica_df = pd.DataFrame(new_selected_features_scaled, columns=['x', 'y'])
        new_ica_df['Country_Region'] = country_region
        
        plt.figure(figsize=(16, 8))
        plt.scatter(new_ica_df['x'], new_ica_df['y'], c='blue', alpha=0.5)
        
        for i, row in new_ica_df.iterrows():
            plt.text(row['x'], row['y'], row['Country_Region'], fontsize=8)
            
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Scatter Plot - {week}')
        
        scatterplot_save_path = os.path.join(scatterplot_save_dir, f'scatterplot_{week}_ICA.png')
        plt.savefig(scatterplot_save_path)
        plt.show()
        
if __name__ == '__main__':
    create_scatterplots(input_dir, output_dir)