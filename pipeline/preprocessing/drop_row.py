import pandas as pd

def drop_row_script(df):
    df.dropna(subset=['Lat', 'Long_'], inplace=True)
        
    return df
