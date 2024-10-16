import warnings
warnings.filterwarnings("ignore")
from keras.models import Model, save_model
from keras.layers import Input, LSTM, RepeatVector
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import tensorflow as tf

source_dir = 'azure-ml-internal-project/data/aggregated'
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

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


timesteps = 1
input_dim = data_scaled.shape[1]
data_scaled_reshaped = data_scaled.reshape(-1, timesteps, input_dim)

X_train, X_test, y_train, y_test = train_test_split(data_scaled_reshaped, data_scaled_reshaped, test_size=0.25, random_state=42)

def recurrent_autoencoder(input_dim, timesteps, latent_dim):
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    autoencoder = Model(inputs, decoded)
    return autoencoder

latent_dim = 32

rae_autoencoder = recurrent_autoencoder(input_dim, timesteps, latent_dim)

num_epochs = 1000
batch_size = 128

rae_autoencoder.compile(optimizer='adam', loss='mse')

history = rae_autoencoder.fit(
    X_train, X_train,
    epochs=num_epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_test, X_test)
)

autoencoder_save_path = r'azure-ml-internal-project\pipeline\training\models'
rae_save = os.path.join(autoencoder_save_path, 'recurrent_ae_model.keras')
save_model(rae_autoencoder, rae_save)
print("Recurrent Autoencoder model saved.")

encoder_model = Model(inputs=rae_autoencoder.input, outputs=rae_autoencoder.layers[1].output)
encoder_save = os.path.join(autoencoder_save_path, 'recurrent_ae_encoder.keras')
save_model(encoder_model, encoder_save)
print("Encoder model saved.")
