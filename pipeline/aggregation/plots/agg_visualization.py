import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

source_dir = r'azure-ml-internal-project\data\aggregated'
destin_dir = r'azure-ml-internal-project\pipeline\aggregation\plots'

csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

dfs = pd.DataFrame()

for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    temp_dfs = pd.read_csv(filepath)
    
    dfs = dfs._append(temp_dfs, ignore_index=True)
dfs['Year-Week'] = dfs['Year'].astype(str) + '-' + dfs['Week'].astype(str).str.zfill(2)
dfs.drop(['Week', 'Year'], axis=1, inplace=True)
dfs.sort_values(by=['Year-Week'], inplace=True)
dfs.reset_index(inplace=True)


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
        
        plt.plot(country_data['Year-Week'], country_data['Confirmed'], label=country)
    
    plt.yscale('log')
    plt.title(f'Confirmed Cases for Countries (Plot {i+1})')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases (Log Scale)')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(destin_dir, f'plot_{i+1}.png'))
    plt.show()
