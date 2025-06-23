# /d:/Programing/Code/Thames-pollution-detect-IoT/MLModel/save_csv_by_station.py

import pandas as pd
import os

# Load the CSV file
df = pd.read_csv('2.7sin_cos_transformed - Copy.csv')

# Create the output directory if it doesn't exist
output_dir = './station'
os.makedirs(output_dir, exist_ok=True)

# Group by the first column (assuming it's named 'Station')
grouped = df.groupby(df.columns[0])

# Save each group to a separate CSV file
for station_name, group in grouped:
    output_path = os.path.join(output_dir, f'{len(group)}_{station_name}.csv')
    group.to_csv(output_path, index=False)