import warnings
import os
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense
import keras
import pickle
import optuna.integration as opt_prun
import tensorflow as tf
from keras import optimizers

warnings.filterwarnings("ignore")

source_dir = 'azure-ml-internal-project/data/aggregated'
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]
activation_mapping = {
    'relu': 0,
    'sigmoid': 1,
    'tanh': 2
}

activation_names = list(activation_mapping.keys())

dfs = []
for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    temp_df = pd.read_csv(filepath)
    dfs.append(temp_df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.reset_index(inplace=True, drop=True)
country_region = combined_df['Country_Region']
numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
data_numeric = combined_df[numeric_columns]

stand_scaler = preprocessing.MinMaxScaler()
data_scaled = stand_scaler.fit_transform(data_numeric)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, data_scaled, test_size=0.25, random_state=42)

def create_variational_autoencoder(trial):
    input_shape = X_train.shape[1]
    num_layers = trial.suggest_int('num_layers', 5, 30)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01)
    activations_idx = trial.suggest_int('activations_idx', 0, len(activation_mapping) - 1)
    activations = activation_names[activations_idx]
    layers_dim = trial.suggest_int('encoder_layer_dim', 16, 128)
    epochs = 200
    batch_size = trial.suggest_int('batch_size', 32, 128)
    latent_dim = 2

    inputs = Input(shape=(input_shape,))
    encoder_layers = [inputs]
    for _ in range(num_layers):
        encoder_layers.append(Dense(layers_dim, activation='relu')(encoder_layers[-1]))

    z_mean = Dense(latent_dim)(encoder_layers[-1])
    z_log_var = Dense(latent_dim)(encoder_layers[-1])

    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

    decoder_layers = [z]
    for layer in encoder_layers[-2::-1]:
        decoder_layers.append(Dense(layer.shape[-1], activation='relu')(decoder_layers[-1]))

    decoded = Dense(input_shape, activation='sigmoid')(decoder_layers[-1])

    autoencoder = Model(inputs, decoded)

    reconstruction_loss = tf.reduce_mean(tf.square(inputs - decoded))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss

    autoencoder.add_loss(vae_loss)
    pruning_callbacks = opt_prun.TFKerasPruningCallback(trial, 'val_loss')
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=learning_rate))

    encoder_model = Model(inputs, z_mean)

    autoencoder.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None), verbose=0, callbacks=[pruning_callbacks])

    encoded_data = encoder_model.predict(X_train)

    return autoencoder, encoder_model


def objective(trial):
    autoencoder, _ = create_variational_autoencoder(trial)
    y_pred = autoencoder.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

autoencoder_save_path = 'azure-ml-internal-project/pipeline/training/models'
vae_save = os.path.join(autoencoder_save_path, 'best_variational_ae_model.keras')
encoder_save = os.path.join(autoencoder_save_path, 'best_variational_encoder_model.keras')
if os.path.exists(vae_save):
    existing_variational_autoencoder = load_model(vae_save)
else:
    existing_variational_autoencoder = None

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.CmaEsSampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
)
study.optimize(objective, n_trials=5)

best_trial = study.best_trial

if existing_variational_autoencoder is None:
    best_variational_autoencoder, best_variational_encoder = create_variational_autoencoder(study.best_trial)
    best_variational_autoencoder.save(vae_save)
    best_variational_encoder.save(encoder_save)
else:
    existing_variational_encoder = load_model(encoder_save)

    best_variational_autoencoder, best_variational_encoder = create_variational_autoencoder(study.best_trial)

    existing_model_predictions = existing_variational_autoencoder.predict(X_test)
    existing_model_mse = mean_squared_error(y_test, existing_model_predictions)

    best_model_predictions = best_variational_autoencoder.predict(X_test)
    best_model_mse = mean_squared_error(y_test, best_model_predictions)

    if best_model_mse < existing_model_mse:
        best_variational_autoencoder.save(vae_save)
        best_variational_encoder.save(encoder_save)
