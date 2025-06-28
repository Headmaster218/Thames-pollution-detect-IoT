import pandas as pd
import numpy as np
from datetime import datetime

# 读取CSV文件
df = pd.read_csv('2.6useful_data_New.csv')

# 提取第三列的时间数据
time_data = pd.to_datetime(df.iloc[:, 2])

# 计算dayoftheyear的余弦和正弦变换
day_of_year = time_data.dt.dayofyear
cos_day_of_year = np.cos(2 * np.pi * day_of_year / 365.25)
sin_day_of_year = np.sin(2 * np.pi * day_of_year / 365.25)

# 计算timeoftheday的余弦和正弦变换
seconds_in_day = time_data.dt.hour * 3600 + time_data.dt.minute * 60 + time_data.dt.second
cos_time_of_day = np.cos(2 * np.pi * seconds_in_day / 86400)
sin_time_of_day = np.sin(2 * np.pi * seconds_in_day / 86400)

# 将变换后的数据保存到新列
df['cos_day_of_year'] = cos_day_of_year
df['sin_day_of_year'] = sin_day_of_year
df['cos_time_of_day'] = cos_time_of_day
df['sin_time_of_day'] = sin_time_of_day

# 保存修改后的CSV文件
df.to_csv('2.7sin_cos_transformed.csv', index=False)
