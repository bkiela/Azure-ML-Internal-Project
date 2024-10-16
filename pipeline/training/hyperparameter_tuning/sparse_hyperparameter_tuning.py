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

def sparse_autoencoder(input_dim, num_layers, layer_units, learning_rate, activations, sparsity_factor):
    input_layer = Input(shape=(input_dim,))
    
    encoder_layers = [input_layer]
    
    for i in range(num_layers):
        units = layer_units[i] if i < len(layer_units) else layer_units[-1]
        encoder_layers.append(Dense(units, activation=activations)(encoder_layers[-1]))
    
    encoded = Dense(2, activation='linear', activity_regularizer=keras.regularizers.l1(sparsity_factor))(encoder_layers[-1])
    
    decoder_layers = [encoded]
    
    for layer in encoder_layers[-2::-1]:
        decoder_layers.append(Dense(layer.shape[-1], activation=activations)(decoder_layers[-1]))
    
    decoded = Dense(input_dim, activation='sigmoid')(decoder_layers[-1])
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    
    encoder_model = Model(inputs=input_layer, outputs=encoded)
    
    return autoencoder, encoder_model

def objective(trial):
    input_dim = X_train.shape[1]
    num_layers = trial.suggest_int('num_layers', 5, 30)
    layer_units = [trial.suggest_int(f'layer_{i}_units', 4, 128) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01)
    activations_idx = trial.suggest_int('activations_idx', 0, len(activation_mapping) - 1)
    activations = activation_names[activations_idx]
    sparsity_factor = trial.suggest_float('sparsity_factor', 0.001, 0.01)

    sparse_autoencoder_model, encoder_model = sparse_autoencoder(input_dim, num_layers, layer_units, learning_rate, activations, sparsity_factor)

    num_epochs = 200
    batch_size = 128

    history = sparse_autoencoder_model.fit(
        X_train, X_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_test, X_test),
        verbose=0
    )

    y_pred = sparse_autoencoder_model.predict(X_test)
    mse = mean_squared_error(X_test, y_pred)

    return mse

autoencoder_save_path = 'azure-ml-internal-project/pipeline/training/models'
sparse_save = os.path.join(autoencoder_save_path, 'best_sparse_ae_model.keras')
encoder_save = os.path.join(autoencoder_save_path, 'best_sparse_encoder_model.keras')

if os.path.exists(sparse_save):
    existing_sparse_autoencoder = load_model(sparse_save)
else:
    existing_sparse_autoencoder = None

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
)

study.optimize(objective, n_trials=5)

best_trial = study.best_trial
best_hyperparameters = best_trial.params
layer_units = [best_hyperparameters[f'layer_{i}_units'] for i in range(best_hyperparameters['num_layers'])]

best_sparse_autoencoder, best_sparse_encoder_model = sparse_autoencoder(
    input_dim=X_train.shape[1],
    num_layers=best_hyperparameters['num_layers'],
    layer_units=layer_units,
    learning_rate=best_hyperparameters['learning_rate'],
    activations=activation_names[best_hyperparameters['activations_idx']],
    sparsity_factor=best_hyperparameters['sparsity_factor']
)

best_num_epochs = 200
best_batch_size = 128
best_history = best_sparse_autoencoder.fit(
    X_train, X_train,
    epochs=best_num_epochs,
    batch_size=best_batch_size,
    validation_data=(X_test, X_test),
    verbose=0
)

best_y_pred = best_sparse_autoencoder.predict(X_test)
best_mse = mean_squared_error(X_test, best_y_pred)

if existing_sparse_autoencoder is not None:
    existing_y_pred = existing_sparse_autoencoder.predict(X_test)
    existing_mse = mean_squared_error(X_test, existing_y_pred)

    if best_mse < existing_mse:
        save_model(best_sparse_autoencoder, sparse_save)
        save_model(best_sparse_encoder_model, encoder_save)
else:
    save_model(best_sparse_autoencoder, sparse_save)
    save_model(best_sparse_encoder_model, encoder_save)
