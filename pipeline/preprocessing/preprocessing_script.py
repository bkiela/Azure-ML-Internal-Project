import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from drop_scr import * # This script drops the collumns that are not needed for the analysis
from agg_script import * # This script aggregates the data by country
from drop_row import * # This script drops the rows that do not contain any data in Lat and Long columns
from null_fill import * # This script fills the null values in the data
from split_script import * # This script splits the data into 3 groups based on the country names

def main():

    source_dir = os.path.join(os.getcwd(), r'.\..\..\data\raw')
    dataframes = drop_script(source_dir)
    aggregated_dataframes = aggregate_script(dataframes)

    output_dir_base = os.path.join(os.getcwd(),r'.\..\..\data\preprocessed')
    output_subfolders = ['train', 'update_1', 'update_2']
    num_groups = len(output_subfolders)
    
    output_dirs = [os.path.join(output_dir_base, subfolder) for subfolder in output_subfolders]

    all_countries = set()
    for df in aggregated_dataframes:
        all_countries.update(df.index)
    country_groups = split_script(all_countries, num_groups)

    for i, df in tqdm(enumerate(aggregated_dataframes),desc='Saving files'):
        filenames = [filename for filename in os.listdir(source_dir) if filename.endswith('.csv') and '2020' not in filename]
        filename = filenames[i]
        
        df = drop_row_script(df)
        df = null_fill(df)
    
        for subfolder, output_subfolder in zip(output_dirs, output_subfolders):
            groups = country_groups[output_subfolders.index(output_subfolder)]
            df_filtered = df[df.index.isin(groups)]
            
            output_file_path = os.path.join(subfolder, filename)
            df_filtered.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()
