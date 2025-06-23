# /d:/Programing/Code/Thames-pollution-detect-IoT/MLModel/plot_time_distribution.py

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('2.7sin_cos_transformed.csv')

# Extract the third column (time)
time_data = df.iloc[:, 2]

# Convert the time data to datetime format
time_data = pd.to_datetime(time_data)

# Plot the time distribution
plt.figure(figsize=(10, 6))
plt.hist(time_data, bins=50, alpha=0.75)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Time Distribution')
plt.grid(True)
plt.show()