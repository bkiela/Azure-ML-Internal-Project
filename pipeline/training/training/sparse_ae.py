import os
import pandas as pd
import numpy as np
from keras.layers import Dense, Input
from keras import Model
from keras.regularizers import l1
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

source_dir = 'azure-ml-internal-project/data/aggregated'
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

dfs = []
for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    temp_df = pd.read_csv(filepath)
    dfs.append(temp_df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.reset_index(inplace=True, drop=True)
numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
data_numeric = combined_df[numeric_columns]

stand_scaler = preprocessing.MinMaxScaler()
data_scaled = stand_scaler.fit_transform(data_numeric)

def sparse_autoencoder(input_shape, dim_num, sparsity_factor):
    inputs = Input(shape=input_shape)
    
    encoder = Dense(128, activation='relu')(inputs)
    encoder = Dense(64, activation='relu')(encoder)
    encoded = Dense(dim_num, activation='linear',
                    activity_regularizer=l1(sparsity_factor))(encoder)
    
    decoder = Dense(64, activation='relu')(encoded)
    decoder = Dense(128, activation='relu')(decoder)
    decoded = Dense(input_shape[0], activation='sigmoid')(decoder)
    
    autoencoder = Model(inputs, decoded)
    encoder_model = Model(inputs, encoded)
    
    return autoencoder, encoder_model

input_shape = (data_scaled.shape[1],)
dim_num = 2
sparsity_factor = 0.01
epochs = 1000
batch_size = 64

autoencoder, encoder_model = sparse_autoencoder(input_shape, dim_num, sparsity_factor)
autoencoder.compile(optimizer='adam', loss='mse')

X_train, X_test, y_train, y_test = train_test_split(data_scaled, data_scaled, test_size=0.25, random_state=42)

autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size)

encoded_data = encoder_model.predict(X_train)
print("Encoded data with 2 dimensions:")
print(encoded_data)

sprs_save = r'azure-ml-internal-project\pipeline\training\models'
sprs_mdl_save = os.path.join(sprs_save, 'sparse_ae_model.keras')
autoencoder.save(sprs_mdl_save)

encoder_save = os.path.join(sprs_save, 'sparse_ae_encoder.keras')
encoder_model.save(encoder_save)
