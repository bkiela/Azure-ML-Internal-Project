import os
import pandas as pd
import pickle as pkl
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler

source_dir = r'azure-ml-internal-project\data\aggregated'
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

dfs = []

for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    temp_df = pd.read_csv(filepath)
    dfs.append(temp_df)

combined_df = pd.concat(dfs, ignore_index=True)
country_region = combined_df['Country_Region']

def sdae_reduction(data, dim_num=2):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data_numeric = data[numeric_columns]
    stand_scaler = StandardScaler()
    data_scaled = stand_scaler.fit_transform(data_numeric)

    layers_dim = [data_numeric.shape[1], 8, 5, dim_num]

    gaussian_encoder = keras.models.Sequential([
        layers.Input(shape=(data_scaled.shape[1],)),
        layers.GaussianNoise(0.3),
        layers.Dense(layers_dim[0], activation='relu'),
        layers.Dense(layers_dim[1], activation='relu'),
        layers.Dense(layers_dim[2], activation='relu'),
        layers.Dense(layers_dim[3], activation='relu')
    ])

    gaussian_decoder = keras.models.Sequential([
        layers.Input(shape=(dim_num,)),
        layers.Dense(layers_dim[2], activation='relu'),
        layers.Dense(layers_dim[1], activation='relu'),
        layers.Dense(layers_dim[0], activation='relu'),
        layers.Dense(data_scaled.shape[1], activation='sigmoid')
    ])

    gaussian_ae = keras.models.Sequential([gaussian_encoder, gaussian_decoder])
    opt = keras.optimizers.Adam(learning_rate=0.001)
    gaussian_ae.compile(loss='mse', optimizer=opt)

    history = gaussian_ae.fit(data_scaled, data_scaled, epochs=1500, batch_size=120, verbose=1)
    encoded_data = gaussian_encoder.predict(data_scaled)

    models_directory = 'azure-ml-internal-project/pipeline/training/models'

    model_filename = os.path.join(models_directory, 'gaussian_autoencoder_model.keras')
    scaler_filename = os.path.join(models_directory, 'Stand_Scaler.pkl')

    gaussian_ae.save(model_filename)
    with open(scaler_filename, 'wb') as scaler_file:
        pkl.dump(stand_scaler, scaler_file)

    return encoded_data

encoded_data = sdae_reduction(combined_df, dim_num=2)

