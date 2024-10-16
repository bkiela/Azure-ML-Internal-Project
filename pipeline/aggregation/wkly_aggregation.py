import pandas as pd
import numpy as np
import os

input_dir = os.path.join(os.getcwd(), r'.\..\..\data\preprocessed\train')
output_dir = os.path.join(os.getcwd(), r'.\..\..\data\aggregated')

csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv') and '2020' not in file]

combined_data = pd.DataFrame()

for filename in csv_files:
    date_str = os.path.splitext(filename)[0]
    date_obj = pd.to_datetime(date_str, format='%m-%d-%Y')
    
    df = pd.read_csv(os.path.join(input_dir, filename))

    week = date_obj.week
    df['Week'] = week
    year = date_obj.year
    df['Year'] = year
    
    combined_data = pd.concat([combined_data, df], ignore_index=True)

grouped = combined_data.groupby(['Country_Region', 'Week', 'Year']).agg({
    'Lat': np.median,
    'Long_': np.median,
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Incident_Rate': np.average,
    'Case_Fatality_Ratio': np.average,
}).reset_index()

for (week, year), group_df in grouped.groupby(['Week', 'Year']):
    if week != 53:
        output_filename = os.path.join(output_dir, f'week_{week}_year_{year}.csv')
        group_df.to_csv(output_filename, index=False)
