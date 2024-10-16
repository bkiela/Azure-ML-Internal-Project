import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import pickle


def create_scatterplots(model_dir):
    input_dir = r'azure-ml-internal-project\data\aggregated'
    output_dir = r'azure-ml-internal-project\pipeline\evaluation\plots'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    with open(model_dir, 'rb') as model_file:
        model = pickle.load(model_file)
    
    scatterplots_save_dir = os.path.join(output_dir)
    last_component = model_dir.split("\\")[-1].split("_")[0]
    
    for week_num in range(25, 31):
        week_num = f'week_{week_num}_year_2022.csv'
        input_csv = os.path.join(input_dir, week_num)
        output_dir = f'encoded_data_{week_num}_{last_component}'
        
        data = pd.read_csv(input_csv)
        country_region = data['Country_Region']
        selected_features = data.drop(['Country_Region'], axis=1)
        
        scaler = StandardScaler()
        selected_features_scaled = scaler.fit_transform(selected_features)
        selected_features_scaled = selected_features_scaled - selected_features_scaled.min() + 0.000001
        
        new_selected_features_scaled = model.transform(selected_features_scaled)
        new_model_df = pd.DataFrame(new_selected_features_scaled, columns=['x', 'y'])
        new_model_df['Country_Region'] = country_region
        
        plt.figure(figsize=(16, 8))
        plt.scatter(new_model_df['x'], new_model_df['y'], c='blue', alpha=0.5)
        
        for i, row in new_model_df.iterrows():
            plt.text(row['x'], row['y'], row['Country_Region'], fontsize=8)
            
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Scatter Plot - {week_num}')
        
        scatterplot_save_path = os.path.join(scatterplots_save_dir, f'scatterplot_{week_num.split(".")[0]}_{last_component}.png')
        plt.savefig(scatterplot_save_path)
        
        
if __name__ == '__main__':
    model_dir = r'azure-ml-internal-project\pipeline\training\models\ica_model.pkl'
    
    create_scatterplots(model_dir)
            