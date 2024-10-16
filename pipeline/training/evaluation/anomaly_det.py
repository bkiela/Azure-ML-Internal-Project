import pandas as pd
import tensorflow as tf
from keras.layers import Input
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import numpy as np

best_loss = float('inf')
best_hyperparameters = None

source_dir = r'azure-ml-internal-project\data\aggregated'
csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

data = []

for filename in csv_files:
    filepath = os.path.join(source_dir, filename)
    temp_df = pd.read_csv(filepath)
    data.append(temp_df)

combined_df = pd.concat(data, ignore_index=True)

random_countries = np.random.choice(combined_df['Country_Region'].unique(), size=5, replace=False)

combined_df = combined_df[combined_df['Country_Region'].isin(random_countries)]

country_region = combined_df['Country_Region']
year = combined_df['Year']
combined_df.drop(['Country_Region'], axis=1, inplace=True)

data_numeric = combined_df.select_dtypes(include=['float64', 'int64'])
data_numeric = data_numeric.astype('float32')

data_tensor = tf.convert_to_tensor(data_numeric.values, dtype=tf.float32)

model = tf.keras.models.load_model(r'azure-ml-internal-project\pipeline\training\models\best_gauss_ae_model.keras')

reconstructions = model.predict(data_tensor)
mse_per_sample = tf.keras.losses.mean_squared_error(data_tensor, reconstructions)

anomaly_scores = pd.Series(mse_per_sample.numpy(), name='anomaly_scores')
anomaly_scores.index = data_numeric.index

df = pd.DataFrame(combined_df)
df['Country_Region'] = country_region
df.drop(df.index[-3:], axis=0, inplace=True)

for country in random_countries:
    country_data_indices = df[df['Country_Region'] == country].index
    country_anomaly_scores = anomaly_scores.loc[country_data_indices]
    
    country_threshold = np.quantile(country_anomaly_scores, 0.95)
    country_anomalous = country_anomaly_scores > country_threshold
    
    df.loc[country_data_indices, 'Anomaly'] = country_anomalous.astype(int)

df['Year-Week'] = df['Year'].astype(str) + '-' + df['Week'].astype(str).str.zfill(2)
df = df[~((df['Year'] == 2021) & (df['Week'] >= 1) & (df['Week'] <= 3))]
df.drop(['Week', 'Year'], axis=1, inplace=True)
df.sort_values(by=['Year-Week'], inplace=True)
df.reset_index(inplace=True)
df.drop(df.index[-3:], axis=0, inplace=True)


true_labels = df['Anomaly']
binary_labels = df['Anomaly']
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, binary_labels, average='binary')


print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1_score}')

plt.figure(figsize=(16, 8))


for country in random_countries:
    country_data = df[df['Country_Region'] == country]
    plt.plot(country_data['Year-Week'], country_data['Incident_Rate'], label=country)
    
    anomalous_country_data = country_data[country_data['Anomaly'] == True]
    plt.scatter(anomalous_country_data['Year-Week'], anomalous_country_data['Incident_Rate'], c='red', marker='o')

plt.title('Incident_Rate Cases for Different Countries')
plt.xlabel('Year-Week')
plt.ylabel('Incident_Rate Cases')
plt.yscale('log')
plt.xticks(rotation=90, size=8)
plt.legend()

if not os.path.exists(r'azure-ml-internal-project\pipeline\training\evaluation\plots'):
    os.mkdir(r'azure-ml-internal-project\pipeline\training\evaluation\plots')
plt.savefig(r'azure-ml-internal-project\pipeline\training\evaluation\plots\anomaly_detection.png')
plt.show()
