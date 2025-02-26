import csv

input_file = '1 2000-2025Coliform.csv'
output_file = '2.2extracted_specific_columns.csv'
columns_to_extract = [
    "Cond @ 25C","Cond 20C lea","Cond - Field","TDS Leach DW", "Oxygen Diss", "O Dissolved","O Diss% Satn", "Temp Water", "pH","pH Inst unit","pH in-situ", "Tot Coli CMF",'Turbidity',"TurbidityNTU","Turbdty in-s","Transparency",
    "E.coli Pres","E.coli Conf", "E.coli C-MF", "Colfm C-MF","ColfmF C-MF","Colfm P-MPN", "E.coli PMF","Colfm PMF","Colfm PMF10","E.coli PMF10","E.coli C-MPN","Ecoli P-MPNB", "ColfmF Conf","Colfrm Conf","Colfm Conf","ColfmF MF10", "ColformsPre", "F Coli Pre","EColi HH2","ColfmF PMF",
]

# 读取输入文件并提取指定列的整行
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    rows_to_write = [row for row in reader if row['determinand.label'] in columns_to_extract]

# 将提取的行写入输出文件
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows_to_write)
