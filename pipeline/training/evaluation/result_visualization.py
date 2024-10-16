import pandas as pd
import matplotlib.pyplot as plt
import os

def create_scatterplots(input_dir, output_dir):
    scatterplot_save_dir = os.path.join(output_dir)

    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    for csv_file in csv_files:
        data = pd.read_csv(os.path.join(input_dir, csv_file))

        plt.figure(figsize=(16, 8))
        plt.scatter(data['x'], data['y'], c='blue', alpha=0.7)

        for i, row in data.iterrows():
            plt.text(row['x'], row['y'], row['Country_Region'], fontsize=8)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Scatter Plot - {csv_file}')

        scatterplot_save_path = os.path.join(scatterplot_save_dir, f'scatterplot_{csv_file}.png')
        plt.savefig(scatterplot_save_path)
        plt.show()
