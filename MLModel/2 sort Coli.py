import os
import pandas as pd

# Read the CSV file
df = pd.read_csv('2000-2025Coliform.csv')

# Create a directory to store station data
output_dir = './station/'
os.makedirs(output_dir, exist_ok=True)

# Group by station name
grouped = df.groupby(df.columns[0])

for station, group in grouped:
    # Get the number of unique timestamps
    unique_timestamps = group.iloc[:, 2].nunique()
    
    # Construct the filename
    filename = f"{unique_timestamps}_{station}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save the station data to a file
    group.to_csv(filepath, index=False)
