import csv
from collections import defaultdict

input_file = '2.2extracted_specific_columns.csv'
output_file = '2.3rearranged_columns_Full.csv'
columns_to_extract = [
    "Cond @ 25C","Cond 20C lea","Cond - Field","TDS Leach DW", "Oxygen Diss", "O Dissolved","O Diss% Satn", "Temp Water", "pH","pH Inst unit","pH in-situ", 'Turbidity',"TurbidityNTU","Turbdty in-s","Transparency",
    "Tot Coli CMF","E.coli Pres","E.coli Conf", "E.coli C-MF", "Colfm C-MF","ColfmF C-MF","Colfm P-MPN", "E.coli PMF","Colfm PMF","Colfm PMF10","E.coli PMF10","E.coli C-MPN","Ecoli P-MPNB", "ColfmF Conf","Colfrm Conf","Colfm Conf","ColfmF MF10", "ColformsPre", "F Coli Pre","EColi HH2","ColfmF PMF",
]

# Read the input file and rearrange the data
data = defaultdict(lambda: defaultdict(dict))
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        time = row['sample.sampleDateTime']
        notation = row['sample.samplingPoint.notation']
        label = row['determinand.label']
        if label in columns_to_extract:
            data[time]['sample.samplingPoint.notation'] = notation
            data[time]['sample.samplingPoint.label'] = row['sample.samplingPoint.label']
            data[time][label] = row['result']

# Write to the output file
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    fieldnames = ['sample.samplingPoint.notation', 'sample.samplingPoint.label', 'sample.sampleDateTime'] + columns_to_extract
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for time, values in data.items():
        row = {
            'sample.samplingPoint.notation': values['sample.samplingPoint.notation'],
            'sample.samplingPoint.label': values['sample.samplingPoint.label'],
            'sample.sampleDateTime': time
        }
        row.update(values)
        writer.writerow(row)
