import csv

input_file = '1 2000-2025Coliform.csv'
output_file = '2.1extracted_columns_all_kinds.csv'

# Read the input file and extract the fourth and corresponding fifth columns
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Skip the header row
    columns = [(row[3], row[4]) for row in reader]  # Extract the fourth and corresponding fifth columns

# Remove duplicate entries
unique_columns = list(set(columns))

# Write the unique values to the output file
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Fourth Column', 'Fifth Column'])  # Write the header row
    for item in unique_columns:
        writer.writerow(item)
