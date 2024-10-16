import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import StandardScaler

input_dir = r'azure-ml-internal-project\data\aggregated'
output_dir = r'azure-ml-internal-project\pipeline\evaluation\plots'
model_dir = r'azure-ml-internal-project\pipeline\training\models'
isomap_model_path = os.path.join(model_dir, 'isomap_model.pkl')

with open(isomap_model_path, 'rb') as isomap_file:
    loaded_isomap = pickle.load(isomap_file)
    
def create_scatterplots(input_dir, output_dir):
    scatterplot_save = os.path.join(output_dir)
    
    for week_num in range(25, 31):
        week = f'week_{week_num}_year_2022.csv'
        input_csv = os.path.join(input_dir, week)
        output_dir = f'encoded_data_{week}_isomap'
        
        data = pd.read_csv(input_csv)
        country_region = data['Country_Region']
        selected_features = data.drop(['Country_Region'], axis=1)
        
        scaler = StandardScaler()
        selected_features_scaled = scaler.fit_transform(selected_features)
        
        
        new_selected_features_scaled = loaded_isomap.transform(selected_features_scaled)
        new_isomap_df = pd.DataFrame(new_selected_features_scaled, columns=['x', 'y'])
        new_isomap_df['Country_Region'] = country_region
        
        plt.figure(figsize = (16, 8))
        plt.scatter(new_isomap_df['x'], new_isomap_df['y'], c='blue', alpha=0.5)
        
        for i, row in new_isomap_df.iterrows():
            plt.text(row['x'], row['y'], row['Country_Region'], fontsize=8) 
            
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Scatter Plot - {week}')
        
        scatterplot_save_path = os.path.join(scatterplot_save, f'scatterplot_{week}_isomap.png')
        plt.savefig(scatterplot_save_path)
        plt.show()
        
if __name__ == '__main__':
    create_scatterplots(input_dir, output_dir)

    