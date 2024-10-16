import warnings
import os
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Attention
import keras

warnings.filterwarnings("ignore")

source_dir = 'azure-ml-internal-project/data/aggregated'
autoencoder_save_path = 'azure-ml-internal-project/pipeline/training/models'
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]
activation_mapping = {
    'relu': 0,
    'sigmoid': 1,
    'tanh': 2
}

activation_names = list(activation_mapping.keys())

dfs = [pd.read_csv(os.path.join(source_dir, filename)) for filename in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.reset_index(inplace=True, drop=True)
country_region = combined_df['Country_Region']
numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
data_numeric = combined_df[numeric_columns]

stand_scaler = preprocessing.MinMaxScaler()
data_scaled = stand_scaler.fit_transform(data_numeric)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, data_scaled, test_size=0.25, random_state=42)

def attention_ae(input_dim, num_layers, dim_num, learning_rate, activations, layer_units):
    input_layer = Input(shape=(input_dim,))
    
    encoder_layers = [input_layer]
    
    for i in range(num_layers):
        units = layer_units[i] if i < len(layer_units) else layer_units[-1]
        encoder_layers.append(Dense(units, activation=activations)(encoder_layers[-1]))
    
    encoded = Dense(dim_num, activation='linear')(encoder_layers[-1])
    
    attention_mul = Attention()([encoded, encoded])
    
    encoded_dim = Dense(dim_num, activation='linear')(attention_mul)
    
    decoder_layers = [encoded_dim]
    
    for layer in encoder_layers[-2::-1]:
        decoder_layers.append(Dense(layer.shape[-1], activation=activations)(decoder_layers[-1]))
    
    decoded = Dense(input_dim, activation='sigmoid')(decoder_layers[-1])
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    
    return autoencoder

def objective(trial):
    input_dim = X_train.shape[1]
    num_layers = trial.suggest_int('num_layers', 5, 30)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01)
    activations_idx = trial.suggest_int('activations_idx', 0, len(activation_mapping) - 1)
    activations = activation_names[activations_idx]
    dim_num = 2
    
    layer_units = [trial.suggest_int(f'layer_{i}_units', 4, 128) for i in range(num_layers)]
    
    attention_autoencoder = attention_ae(input_dim, num_layers, dim_num, learning_rate, activations, layer_units)
    
    num_epochs = 100
    batch_size = 128

    history = attention_autoencoder.fit(
        X_train, X_train,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=0
    )

    y_pred = attention_autoencoder.predict(X_test)
    mse = mean_squared_error(X_test, y_pred)

    return mse

autoencoder_save_path = 'azure-ml-internal-project/pipeline/training/models'
att_save = os.path.join(autoencoder_save_path, 'best_attention_ae_model.keras')
encoder_save = os.path.join(autoencoder_save_path, 'best_attention_ae_encoder.keras')

if os.path.exists(att_save) and os.path.exists(encoder_save):
    existing_autoencoder = load_model(att_save)
    existing_encoder = load_model(encoder_save)
else:
    existing_autoencoder = None
    existing_encoder = None

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)

study.optimize(objective, n_trials=10)

best_trial = study.best_trial
best_hyperparameters = best_trial.params

best_attention_autoencoder = attention_ae(
    input_dim=X_train.shape[1],
    num_layers=best_hyperparameters['num_layers'],
    dim_num=2,
    learning_rate=best_hyperparameters['learning_rate'],
    activations=activation_names[best_hyperparameters['activations_idx']],
    layer_units=[best_hyperparameters[f'layer_{i}_units'] for i in range(best_hyperparameters['num_layers'])]
)

best_num_epochs = 100
best_batch_size = 128
best_history = best_attention_autoencoder.fit(
    X_train, X_train,
    epochs=best_num_epochs,
    batch_size=best_batch_size,
    validation_data=(X_test, X_test),
    verbose=0
)

best_y_pred = best_attention_autoencoder.predict(X_test)
best_mse = mean_squared_error(X_test, best_y_pred)

if existing_autoencoder is not None:
    existing_y_pred = existing_autoencoder.predict(X_test)
    existing_mse = mean_squared_error(X_test, existing_y_pred)

    if best_mse < existing_mse:
        save_model(best_attention_autoencoder, att_save)
        save_model(Model(inputs=best_attention_autoencoder.input, outputs=best_attention_autoencoder.layers[-2].output), encoder_save)
else:
    save_model(best_attention_autoencoder, att_save)
    save_model(Model(inputs=best_attention_autoencoder.input, outputs=best_attention_autoencoder.layers[-2].output), encoder_save)
    