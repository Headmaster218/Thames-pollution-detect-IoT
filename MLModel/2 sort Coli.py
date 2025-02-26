import os
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('2000-2025Coliform.csv')

# 创建存储站点数据的目录
output_dir = './station/'
os.makedirs(output_dir, exist_ok=True)

# 根据站点名分组
grouped = df.groupby(df.columns[0])

for station, group in grouped:
    # 获取唯一时间戳的数量
    unique_timestamps = group.iloc[:, 2].nunique()
    
    # 构建文件名
    filename = f"{unique_timestamps}_{station}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # 保存站点数据到文件
    group.to_csv(filepath, index=False)
