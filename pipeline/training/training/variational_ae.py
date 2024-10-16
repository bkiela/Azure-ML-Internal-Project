import warnings
warnings.filterwarnings("ignore")
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
import pandas as pd
import os

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

def variational_autoencoder(input_shape, encoding_dim, dim_num):
    inputs = Input(shape=input_shape)
    encoder = Dense(128, activation='relu')(inputs)
    encoder = Dense(64, activation='relu')(encoder)
    z_mean = Dense(dim_num)(encoder)
    z_log_var = Dense(dim_num)(encoder)
    
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], dim_num),
                              mean=0.0, stddev=1.0)
    z = z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    decoder = Dense(64, activation='relu')(z)
    decoder = Dense(128, activation='relu')(decoder)
    decoded = Dense(input_shape[0], activation='sigmoid')(decoder)
    
    autoencoder = Model(inputs, decoded)
    
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - decoded))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss
    
    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer='adam')
    
    encoder_model = Model(inputs, z_mean)
    
    return autoencoder, encoder_model

input_shape = (data_scaled.shape[1],)
encoding_dim = 64
dim_num = 2
epochs = 1000
batch_size = 64

autoencoder, encoder_model = variational_autoencoder(input_shape, encoding_dim, dim_num)

autoencoder.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None))

var_dir = r'azure-ml-internal-project/pipeline/training/models'
var_ae_dir = os.path.join(var_dir, 'variational_ae_model.keras')

autoencoder.save(var_ae_dir)
encoder_save_path = os.path.join(var_dir, 'variational_ae_encoder.keras')
encoder_model.save(encoder_save_path)

encoded_data = encoder_model.predict(X_train)
print("Encoded Data:")
print(encoded_data)
