import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def aggregate_script(dataframes):
    aggregated_dataframes = []

    for df in tqdm(dataframes,desc='Preprocessing'):
        grouped = df.groupby('Country_Region').agg({
            'Lat': np.median,
            'Long_': np.median,
            'Confirmed': 'sum',
            'Deaths': 'sum',
            'Incident_Rate': np.average,
            'Case_Fatality_Ratio': np.average
        })
        
        aggregated_dataframes.append(grouped.reset_index())

    return aggregated_dataframes