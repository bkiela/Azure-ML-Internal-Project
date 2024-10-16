from csv_per_file import sdae_reduction
from result_visualization import create_scatterplot
import os

model_path = r'azure-ml-internal-project\pipeline\training\models\best_gauss_ae_model_opt_encoder.keras'
input_dir = r'azure-ml-internal-project\pipeline\evaluation\csv'
output_dir = r'azure-ml-internal-project\pipeline\evaluation\plots'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

scatter_data = []

for n in range(25, 36):
    week_number = n
    week = f'week_{week_number}_year_2022.csv'
    output_csv = f'encoded_data_{week}'
    input_data = fr'azure-ml-internal-project\data\aggregated\{week}'

    encoded_data, country_region = sdae_reduction(input_data, model_path, dim_num=2, output_csv=output_csv)
    scatter_data.append((encoded_data, country_region, week))


output_file = 'interactive_scatterplot.html'

if __name__ == '__main__':
    create_scatterplot(scatter_data, output_file)
