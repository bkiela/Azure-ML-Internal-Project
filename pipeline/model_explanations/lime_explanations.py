import os
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from lime import lime_tabular
import matplotlib.pyplot as plt

def prepare_data(df):
    temp_df = df.drop(['Country_Region', 'Year'], axis=1)
    data_numeric = temp_df.select_dtypes(include=['float64', 'int64'])
    data_numeric = data_numeric.astype('float32')
    return tf.convert_to_tensor(data_numeric.values, dtype=tf.float32)


def fit(X):
    data_tensor = stand_scaler.transform(X)
    reconstructions = gaussian_ae.predict(data_tensor)
    mse_per_sample = tf.keras.losses.mean_squared_error(data_tensor, reconstructions).numpy().flatten()
    # threshold = np.quantile(mse_per_sample, 0.90)
    # mse_per_sample -= threshold
    mse_normalized = 1/(1 + np.exp(-mse_per_sample))
    proba = np.vstack([1-mse_normalized, mse_normalized]).T
    return proba


# load model
scaler_filename = os.path.join(os.getcwd(), r'.\..\training\models\best_minmax_scaler.pkl')
model_filename = os.path.join(os.getcwd(), r'.\..\training\models\best_gaussian_autoencoder_model.keras')

stand_scaler = pickle.load(open(scaler_filename,'rb'))
gaussian_ae = tf.keras.models.load_model(model_filename)

source_dir = os.path.join(os.getcwd(),r'.\..\..\data\aggregated')
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

data = []

for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    temp_df = pd.read_csv(filepath)
    data.append(temp_df)

df = pd.concat(data, ignore_index=True)

reference_countries = ['Fiji','France','Sweden']
test_countries = ['Russia','Slovakia','US']

reference_data = df.loc[df['Country_Region'].isin(reference_countries)]
test_data = df.loc[df['Country_Region'].isin(test_countries)]
reference_tensor = prepare_data(reference_data)
test_tensor = prepare_data(test_data)

score = fit(test_tensor)

explainer = lime_tabular.LimeTabularExplainer(
    training_data=test_tensor.numpy(),
    feature_names=test_data.drop(['Country_Region', 'Year'], axis=1).columns,
    class_names=['Normal', 'Anomaly'],
    mode='classification'
)

exp = explainer.explain_instance(
    data_row=test_data.drop(['Country_Region', 'Year'], axis=1).iloc[1],
    predict_fn=fit
)

exp.as_pyplot_figure()
plt.savefig('./plots/lime_explanation.png')
plt.show()