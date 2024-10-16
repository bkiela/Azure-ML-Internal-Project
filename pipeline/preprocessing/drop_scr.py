import os
import pandas as pd
from tqdm.auto import tqdm

def drop_script(source_dir):
    modified_dataframes = []

    for filename in tqdm(os.listdir(source_dir),desc='Loading files'):
        if filename.endswith('.csv') and '2020' not in filename:
            file_path = os.path.join(source_dir, filename)
            df = pd.read_csv(file_path)

            columns_to_drop = ['FIPS', 'Admin2', 'Province_State', 'Last_Update', 'Recovered', 'Active', 'Combined_Key']
            df = df.drop(columns=columns_to_drop)

            modified_dataframes.append(df)

    return modified_dataframes

