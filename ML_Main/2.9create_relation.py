import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('2.7sin_cos_transformed - Copy.csv')

# 获取需要处理的列
columns_to_process = df.columns[3:8].tolist() + df.columns[9:12].tolist()

# 将列转换为浮点数
for col in columns_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 生成新的列
for i in range(len(columns_to_process)):
    for j in range(i + 1, len(columns_to_process)):
        col1 = columns_to_process[i]
        col2 = columns_to_process[j]
        df[f'{col1} * {col2}'] = df[col1] * df[col2]
        # df[f'log({col1} / {col2})'] = np.log(df[col1] / df[col2] + 1)
        # df[f'log({col2} / {col1})'] = np.log(df[col2] / df[col1] + 1)

# 生成单一变量列
for col in columns_to_process:
    df[f'{col} n/(n+1)'] = df[col] / (df[col] + 1)

# 保存处理后的数据
df.to_csv('2.7sin_cos_transformed - Copy.csv', index=False)
