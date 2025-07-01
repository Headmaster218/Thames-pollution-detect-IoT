import pandas as pd
import numpy as np
from datetime import datetime

# Read the CSV file
df = pd.read_csv('2.6useful_data_New.csv')

# Extract time data from the third column
time_data = pd.to_datetime(df.iloc[:, 2])

# Compute cosine and sine transformations for day of the year
day_of_year = time_data.dt.dayofyear
cos_day_of_year = np.cos(2 * np.pi * day_of_year / 365.25)
sin_day_of_year = np.sin(2 * np.pi * day_of_year / 365.25)

# Compute cosine and sine transformations for time of the day
seconds_in_day = time_data.dt.hour * 3600 + time_data.dt.minute * 60 + time_data.dt.second
cos_time_of_day = np.cos(2 * np.pi * seconds_in_day / 86400)
sin_time_of_day = np.sin(2 * np.pi * seconds_in_day / 86400)

# Save the transformed data into new columns
df['cos_day_of_year'] = cos_day_of_year
df['sin_day_of_year'] = sin_day_of_year
df['cos_time_of_day'] = cos_time_of_day
df['sin_time_of_day'] = sin_time_of_day

# Save the modified CSV file
df.to_csv('2.7sin_cos_transformed.csv', index=False)
