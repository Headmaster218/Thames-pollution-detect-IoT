import csv

input_file = '1 2000-2025Coliform.csv'
output_file = '2.1extracted_columns_all_kinds.csv'

# 读取输入文件并提取第四列和对应的第五列
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # 跳过标题行
    columns = [(row[3], row[4]) for row in reader]  # 提取第四列和对应的第五列

# 删除重复内容
unique_columns = list(set(columns))

# 将唯一值写入输出文件
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Fourth Column', 'Fifth Column'])  # 写入标题行
    for item in unique_columns:
        writer.writerow(item)
