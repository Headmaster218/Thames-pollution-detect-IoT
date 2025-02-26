import pandas as pd

# 读取CSV文件
df = pd.read_csv('2.4 hand merge Cond @ 25COxygen DissTemp Turbidity.csv')

# 检查第4到第8列的有效数据数量
valid_data_count = df.iloc[:, 3:8].notna().sum(axis=1)

# 删除有效数据少于5的行
filtered_df = df[valid_data_count >= 5]

# 保存处理后的数据到一个新的CSV文件
filtered_df.to_csv('2.5filtered_data.csv', index=False)
