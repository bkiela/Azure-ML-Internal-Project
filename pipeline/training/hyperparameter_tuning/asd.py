import asyncio
import concurrent.futures
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import os
import pandas as pd
import IPython.display as display
import numpy as np
from itertools import product
from sklearn import preprocessing
from tensorflow import keras
from keras import layers

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
combined_df.drop(['Country_Region'], axis=1, inplace=True)
numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
data_numeric = combined_df[numeric_columns]

stand_scaler = preprocessing.MinMaxScaler()
data_scaled = stand_scaler.fit_transform(data_numeric)

def train_autoencoder(hyperparameters, data_scaled):
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

    encoded_layer = layers.Dense(2, activation=activation_ch)(encoder_output)
    encoder = keras.models.Model(encoder_input, encoded_layer)

    decoder_input = layers.Input(shape=(2,))
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

    loss_values = []
    epoch_numbers = []

    fig, ax = plt.subplots()

    for epoch in range(epoch_num):
        history = autoencoder.fit(
            data_scaled,
            data_scaled,
            epochs=1,
            batch_size=batch_size_num,
            verbose=1
        )

        loss_values.append(history.history['loss'][0])
        epoch_numbers.append(epoch + 1)

        ax.clear()
        ax.plot(epoch_numbers, loss_values, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.pause(0.001)

        plt.cla()
        plt.draw()

        if best_hyperparameters is None or history.history['loss'][0] < best_loss:
            best_loss = history.history['loss'][0]
            best_hyperparameters = {
                'num_layers_values': num_layers_values,
                'layer_dim_values': layer_dim_values,
                'activation': activation_ch,
                'epochs': epoch + 1,
                'batch_size': batch_size_num,
                'learning_rate': learning_rate_num
            }
            models_directory = r'azure-ml-internal-project\pipeline\training\models'

            model_filename = os.path.join(models_directory, 'best_ae_model.keras')

            autoencoder.save(model_filename)

    display.clear_output()

def save_results_async(encoded_data, best_loss, best_hyperparameters, country_region):
    encoded_df = pd.DataFrame(encoded_data, columns=['x', 'y'])
    encoded_df['Country_Region'] = country_region

    encoded_csv_directory = r'azure-ml-internal-project\pipeline\training\hyperparameter_tuning'

    encoded_csv_path = os.path.join(encoded_csv_directory, 'encoded_data.csv')
    encoded_df.to_csv(encoded_csv_path, index=False)

def train_autoencoders_parallel(hyperparameter_grid, data_scaled, country_region):
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        loop = asyncio.get_event_loop()

        async def train_and_save(hyperparameters):
            encoded_data, best_loss, best_hyperparameters = await loop.run_in_executor(
                executor, partial(train_autoencoder, hyperparameters, data_scaled)
            )
            await save_results_async(encoded_data, best_loss, best_hyperparameters, country_region)

        tasks = [train_and_save(hp) for hp in hyperparameter_grid]
        loop.run_until_complete(asyncio.gather(*tasks))

async def main():
    hyperparameter_grid = product(
        [3, 4, 5, 6, 7, 8, 9, 10],
        [10, 15, 20, 25, 30],
        ["relu", "sigmoid", "tanh"],
        range(100, 1001, 100),
        [128],
        np.linspace(0.0001, 0.1, num=20)
    )

    await train_autoencoders_parallel(hyperparameter_grid, data_scaled, country_region)

if __name__ == '__main__':
    asyncio.run(main())
