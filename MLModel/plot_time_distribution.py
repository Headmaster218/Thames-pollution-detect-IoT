
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('2.7sin_cos_transformed.csv')

# Extract the third column (time)
time_data = df.iloc[:, 2]

# Convert the time data to datetime format
time_data = pd.to_datetime(time_data)

# Plot each time as a point on a line
plt.figure(figsize=(10, 6))
plt.plot(time_data, [1] * len(time_data), '|', alpha=0.75)
plt.xlabel('Time')
plt.title('Time Points Distribution')
plt.grid(True)
plt.show()
