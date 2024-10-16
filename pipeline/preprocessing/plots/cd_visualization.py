import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

source_dir = r'azure-ml-internal-project\data\preprocessed\train'
destin_dir = r'azure-ml-internal-project\pipeline\preprocessing\plots'

csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

dfs = pd.DataFrame()

for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    date_str = os.path.splitext(filename)[0]
    date_obj = pd.to_datetime(date_str, format='%m-%d-%Y')
    temp_df = pd.read_csv(filepath)
    temp_df['Date'] = date_obj
    
    dfs = dfs._append(temp_df, ignore_index=True)
    dfs.sort_values(by=['Date', 'Country_Region'], inplace=True)

unique_countries = dfs['Country_Region'].unique()

countries_per_plot = 5
num_plots = len(unique_countries) // countries_per_plot

for i in range(num_plots):
    if dfs['Confirmed'].sum() == 0:
        continue 
    start_idx = i * countries_per_plot
    end_idx = (i + 1) * countries_per_plot
    countries_subset = unique_countries[start_idx:end_idx]
    
    plt.figure(figsize=(12, 8))
    
    for country in countries_subset:
        country_data = dfs[dfs['Country_Region'] == country]
        
        if country_data['Confirmed'].sum() == 0:
            continue
        
        plt.plot(country_data['Date'], country_data['Confirmed'], label=country)
    
    plt.yscale('log')
    plt.title(f'Confirmed Cases for Countries (Plot {i+1})')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases (Log Scale)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(destin_dir, f'plot_{i+1}.png'))
    plt.show()
