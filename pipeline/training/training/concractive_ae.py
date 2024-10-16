import warnings
warnings.filterwarnings("ignore")
from keras.models import Model, save_model
from keras.layers import Input, Dense
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

X_train, X_test, y_train, y_test = train_test_split(data_scaled, data_scaled, test_size=0.25, random_state=42)

def contractive_ae(dim_num=2, lambda_reg=0.001):
    input_dim = X_train.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(dim_num)(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    jacobian = Model(inputs=input_layer, outputs=encoded)
    jacobian.compile(optimizer='adam', loss='mse')

    def frobenius_norm(y_true, y_pred):
        return tf.norm(jacobian.input[0], ord='euclidean') ** 2

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.add_loss(lambda_reg * frobenius_norm(None, None))
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

contractive_autoencoder = contractive_ae(dim_num=2)

num_epochs = 1000
batch_size = 128

history = contractive_autoencoder.fit(
    X_train, X_train,
    epochs=num_epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_test, X_test)
)

autoencoder_save_path = r'azure-ml-internal-project\pipeline\training\models'
contractive_ae_save = os.path.join(autoencoder_save_path, 'contractive_ae_model.keras')
save_model(contractive_autoencoder, contractive_ae_save)
print("Contractive Autoencoder model saved.")

encoder_model = Model(inputs=contractive_autoencoder.input, outputs=contractive_autoencoder.layers[2].output)
encoder_save = os.path.join(autoencoder_save_path, 'contractive_ae_encoder.keras')
save_model(encoder_model, encoder_save)
print("Encoder model saved.")
