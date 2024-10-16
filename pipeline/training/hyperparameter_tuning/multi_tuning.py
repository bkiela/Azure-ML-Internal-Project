from tensorflow import keras
from keras import layers
from sklearn import preprocessing
import pandas as pd
import pickle as pkl
import os
import numpy as np
from keras.callbacks import EarlyStopping
from itertools import product
import asyncio
import concurrent.futures

best_loss = float('inf')
best_hyperparameters = None

source_dir = r'azure-ml-internal-project\data\aggregated'
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

dfs = []

for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    temp_df = pd.read_csv(filepath)
    dfs.append(temp_df)

combined_df = pd.concat(dfs, ignore_index=True)
country_region = combined_df['Country_Region']
year = combined_df['Year']
combined_df.drop(['Country_Region'], axis=1, inplace=True)
numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
data_numeric = combined_df[numeric_columns]

stand_scaler = preprocessing.MinMaxScaler()
data_scaled = stand_scaler.fit_transform(data_numeric)

def sdae_reduction(data_scaled, dim_num=2, hyperparameters=None):
    global best_loss
    global best_hyperparameters

    num_layers_values, layer_dim_values, activation_ch, epoch_num, batch_size_num, learning_rate_num = hyperparameters

    neurons_per_layer = [
        max(int(layer_dim_values / (2**i)), 2)
        for i in range(num_layers_values)
    ]

    encoder_input = layers.Input(shape=(data_scaled.shape[1],))
    encoder_output = encoder_input
    for dim in neurons_per_layer[:-1]:
        encoder_output = layers.Dense(dim, activation=activation_ch)(encoder_output)

    encoded_layer = layers.Dense(dim_num, activation=activation_ch)(encoder_output)
    encoder = keras.models.Model(encoder_input, encoded_layer)

    decoder_input = layers.Input(shape=(dim_num,))
    decoder_output = decoder_input
    for dim in reversed(neurons_per_layer[:-1]):
        decoder_output = layers.Dense(dim, activation=activation_ch)(decoder_output)
    decoder_output = layers.Dense(data_scaled.shape[1], activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    autoencoder_input = encoder_input
    autoencoder_output = decoder(encoded_layer)
    autoencoder = keras.models.Model(autoencoder_input, autoencoder_output)

    opt = keras.optimizers.Adam(learning_rate=learning_rate_num)
    autoencoder.compile(loss="mse", optimizer=opt)

    history = autoencoder.fit(data_scaled, data_scaled, epochs=epoch_num, batch_size=batch_size_num, verbose=0)
    encoded_data = encoder.predict(data_scaled)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        restore_best_weights=True
    )

    history = autoencoder.fit(
        data_scaled,
        data_scaled,
        epochs=epoch_num,
        batch_size=batch_size_num,
        verbose=0,
        callbacks=[early_stopping],
        validation_split=0.2
    )

    if best_hyperparameters is None or history.history['val_loss'][-1] < best_loss:
        best_loss = history.history['val_loss'][-1]
        best_hyperparameters = {
            'num_layers_values': num_layers_values,
            'layer_dim_values': layer_dim_values,
            'activation': activation_ch,
            'epochs': epoch_num,
            'batch_size': batch_size_num,
            'learning_rate': learning_rate_num
        }
        models_directory = r'azure-ml-internal-project/pipeline/training/models'

        model_filename = os.path.join(models_directory, 'best_ae_model.keras')
        scaler_filename = os.path.join(models_directory, 'best_minmax_scaler.pkl')

        autoencoder.save(model_filename)
        with open(scaler_filename, 'wb') as scaler_file:
            pkl.dump(stand_scaler, scaler_file)

    return encoded_data

hyperparameter_grid = product(
        [3, 4, 5, 6, 7, 8, 9, 10],
        [10, 15, 20, 25, 30],
        ["relu", "sigmoid", "tanh"],
        range(100, 1001, 100),
        [128],
        np.linspace(0.0001, 0.1, num=20)
    )

def hyperparameter_tuning(hyperparameters):
    num_layers_values, layer_dim_values, activation_ch, epoch_num, batch_size_num, learning_rate_num = hyperparameters
    print(f"Hyperparameters: {hyperparameters}")

    encoded_data = sdae_reduction(data_scaled, dim_num=2, hyperparameters=hyperparameters)
    print(f'Best Loss: {best_loss}')
    if best_hyperparameters:
        print(f'Best Hyperparameters: {best_hyperparameters}')

async def main():
    tasks = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for hyperparameters in hyperparameter_grid:
            task = loop.run_in_executor(executor, hyperparameter_tuning, hyperparameters)
            tasks.append(task)

        await asyncio.gather(*tasks)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    for hyperparameters in hyperparameter_grid:
        num_layers_values, layer_dim_values, activation_ch, epoch_num, batch_size_num, learning_rate_num = hyperparameters
        print(f"Hyperparameters: {hyperparameters}")

        encoded_data = sdae_reduction(data_scaled, dim_num=2, hyperparameters=hyperparameters)
        print(f'Best Loss: {best_loss}')
        if best_hyperparameters:
            print(f'Best Hyperparameters: {best_hyperparameters}')

    encoded_data, country_region = sdae_reduction(data_scaled, dim_num=2, hyperparameters=best_hyperparameters)

    encoded_df = pd.DataFrame(encoded_data, columns=['x', 'y'])
    encoded_df['Country_Region'] = country_region

    encoded_csv_directory = r'azure-ml-internal-project/pipeline/training/hyperparameter_tuning'

    encoded_csv_path = os.path.join(encoded_csv_directory, 'encoded_data.csv')
    encoded_df.to_csv(encoded_csv_path, index=False)
