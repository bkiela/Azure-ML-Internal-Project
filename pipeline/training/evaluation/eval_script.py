from csv_per_file import sdae_reduction
from result_visualization import create_scatterplots
import os

model_path = r'azure-ml-internal-project\pipeline\training\models\attention_ae_encoder.keras'
input_dir = r'azure-ml-internal-project\pipeline\training\evaluation\csv'
output_dir = r'azure-ml-internal-project\pipeline\training\evaluation\plots'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
    


for n in range(25, 31):
    week_number = n
    week = f'week_{week_number}_year_2022.csv'
    output_csv = f'encoded_data_{week}.csv'
    input_data = fr'azure-ml-internal-project\data\aggregated\{week}'
    
    encoded_data, country_region = sdae_reduction(input_data, model_path, dim_num=2, output_csv=output_csv)

create_scatterplots(input_dir, output_dir)

print("Processing completed.")
 