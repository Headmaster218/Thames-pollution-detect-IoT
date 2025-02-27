import pandas as pd

# Read CSV file
df = pd.read_csv('2.4 hand merge Cond @ 25COxygen DissTemp Turbidity.csv')

# Check the number of valid data in columns 4 to 8
valid_data_count = df.iloc[:, 3:8].notna().sum(axis=1)

# Remove rows with less than 5 valid data points
filtered_df = df[valid_data_count >= 5]

# Save the processed data to a new CSV file
filtered_df.to_csv('2.5filtered_data.csv', index=False)
