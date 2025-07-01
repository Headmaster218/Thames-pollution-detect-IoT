import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('2.7sin_cos_transformed - Copy.csv')

# Get the columns to process
columns_to_process = df.columns[3:8].tolist() + df.columns[9:12].tolist()

# Convert columns to float
for col in columns_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Generate new columns based on pairwise operations
for i in range(len(columns_to_process)):
    for j in range(i + 1, len(columns_to_process)):
        col1 = columns_to_process[i]
        col2 = columns_to_process[j]
        df[f'{col1} * {col2}'] = df[col1] * df[col2]
        # df[f'log({col1} / {col2})'] = np.log(df[col1] / df[col2] + 1)
        # df[f'log({col2} / {col1})'] = np.log(df[col2] / df[col1] + 1)

# Generate single-variable columns
for col in columns_to_process:
    df[f'{col} n/(n+1)'] = df[col] / (df[col] + 1)

# Save the processed data
df.to_csv('2.9sin_cos_with_new_features.csv', index=False)
