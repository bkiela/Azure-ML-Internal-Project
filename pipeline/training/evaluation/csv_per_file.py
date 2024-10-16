from tensorflow import keras
import pandas as pd
import pickle as pkl
import os
from sklearn.preprocessing import MinMaxScaler
from keras import layers, optimizers
from keras.models import load_model

def sdae_reduction(input_data, model_path, dim_num=2, output_csv=None):
    encoder_save = r'azure-ml-internal-project\pipeline\training\evaluation\csv'


    data = pd.read_csv(input_data)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    country_region = data['Country_Region']
    data = data.drop(['Country_Region'], axis=1)
    data_numeric = data[numeric_columns]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    loaded_model = keras.models.load_model(model_path)
    optimizer = optimizers.Adam()
    loaded_model.compile(loss='mean_squared_error', optimizer=optimizer)


    encoded_data = loaded_model.predict(data_scaled)

    if output_csv:
        encoded_df = pd.DataFrame(encoded_data, columns=['x', 'y'])
        encoded_df['Country_Region'] = country_region
        encoder_save = os.path.join(encoder_save, output_csv)
        encoded_df.to_csv(encoder_save, index=False)

    return encoded_data, country_region
