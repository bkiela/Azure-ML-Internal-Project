import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, GaussianNoise
from keras import optimizers
from keras.models import Model, load_model, save_model
import keras
import pickle
import optuna.integration as opt_prun
import optuna

best_model_name = 'best_gauss_ae_model_opt'
best_model_dir = r'azure-ml-internal-project\pipeline\training\models'
best_model_path = os.path.join(best_model_dir, f'{best_model_name}.keras')

if os.path.exists(best_model_path):
    existing_model = load_model(best_model_path)
else:
    existing_model = None

source_dir = 'azure-ml-internal-project/data/aggregated'
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

dfs = [pd.read_csv(os.path.join(source_dir, filename))
       for filename in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.reset_index(inplace=True, drop=True)
country_region = combined_df['Country_Region']
numeric_columns = combined_df.select_dtypes(
    include=['float64', 'int64']).columns
data_numeric = combined_df[numeric_columns]

stand_scaler = MinMaxScaler()
data_scaled = stand_scaler.fit_transform(data_numeric)

X_train, X_test, y_train, y_test = train_test_split(
    data_scaled, data_scaled, test_size=0.25, random_state=42)


def create_neural_network(trial):
    input_shape = X_train.shape[1]
    num_layers = trial.suggest_int('num_layers', 5, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01)
    activation_names = ['relu', 'sigmoid', 'tanh']
    activations_idx = trial.suggest_int(
        'activations_idx', 0, len(activation_names) - 1)
    activations = activation_names[activations_idx]
    layers_dim = trial.suggest_int('encoder_layer_dim', 16, 128)
    dim_num = 2

    encoder_input = Input(shape=(input_shape,))
    encoder_layers = [encoder_input]
    encoder_layers.append(GaussianNoise(0.3)(encoder_layers[-1]))

    for _ in range(num_layers):
        encoder_layers.append(
            Dense(layers_dim, activation=activations)(encoder_layers[-1]))
        layers_dim = max(layers_dim // 2, 4)

    encoder_layers.append(
        Dense(dim_num, activation=activations)(encoder_layers[-1]))
    gaussian_encoder = keras.models.Model(encoder_input, encoder_layers[-1])

    decoder_input = Input(shape=(dim_num,))
    decoder_layers = [decoder_input]

    for layer in encoder_layers[-2::-1]:
        decoder_layers.append(
            Dense(layer.shape[-1], activation=activations)(decoder_layers[-1]))

    decoder_layers.append(
        Dense(input_shape, activation='sigmoid')(decoder_layers[-1]))
    gaussian_decoder = keras.models.Model(decoder_input, decoder_layers[-1])

    autoencoder_input = Input(shape=(input_shape,))
    encoded = gaussian_encoder(autoencoder_input)
    decoded = gaussian_decoder(encoded)
    autoencoder = keras.models.Model(autoencoder_input, decoded)

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    pruning_callbacks = opt_prun.TFKerasPruningCallback(trial, 'val_loss')
    autoencoder.compile(loss='mean_squared_error', optimizer=optimizer)

    autoencoder.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=200, batch_size=128, verbose=0, callbacks=[pruning_callbacks])

    encoded_data = gaussian_encoder.predict(X_train)
    encoded_data_save = os.path.join(
        r'azure-ml-internal-project\pipeline\training\hyperparameter_tuning', 'encoded_data.csv')
    encoded_df = pd.DataFrame(encoded_data, columns=['x', 'y'])
    encoded_df['Country_Region'] = country_region
    encoded_df.to_csv(encoded_data_save, index=False)

    return autoencoder


def objective(trial):
    model = create_neural_network(trial)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.CmaEsSampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
)

study.optimize(objective, n_trials=150)

best_trial = study.best_trial
save_dir = r'azure-ml-internal-project\pipeline\training\models'
existing_scaler_path = os.path.join(save_dir, 'stand_scaler.pkl')

if existing_model:
    existing_model_predictions = existing_model.predict(X_test)
    existing_model_mse = mean_squared_error(y_test, existing_model_predictions)

    best_model = create_neural_network(best_trial)

    best_model_predictions = best_model.predict(X_test)
    best_model_mse = mean_squared_error(y_test, best_model_predictions)

    if best_model_mse < existing_model_mse:
        best_model.save(best_model_path)
        with open(existing_scaler_path, 'wb') as scaler_file:
            pickle.dump(stand_scaler, scaler_file)

        best_encoder = keras.models.Model(
            best_model.layers[1].input, best_model.layers[-2].output)

        best_encoder_path = os.path.join(
            best_model_dir, f'{best_model_name}_encoder.keras')
        best_encoder.save(best_encoder_path)
else:
    best_model = create_neural_network(study.best_trial)
    best_model.save(best_model_path)
    with open(existing_scaler_path, 'wb') as scaler_file:
        pickle.dump(stand_scaler, scaler_file)

    best_encoder = keras.models.Model(
        best_model.layers[1].input, best_model.layers[-2].output)

    best_encoder_path = os.path.join(
        best_model_dir, f'{best_model_name}_encoder.keras')
    best_encoder.save(best_encoder_path)
