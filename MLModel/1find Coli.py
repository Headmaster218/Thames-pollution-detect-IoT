import pandas as pd
import os
from multiprocessing import Pool

def process_file(year):
    filename = 'D:\\Downloads\\' + f'{year}.csv'
    if os.path.exists(filename):
        print(f'Processing {filename}...')
        # Read the CSV file
        df = pd.read_csv(filename)
        # Filter rows where the sixth column is "ColfmF Conf" or "E.coli C-MF"
        coliform_indices = df[(df.iloc[:, 5] == 'ColfmF Conf') | (df.iloc[:, 5] == 'E.coli C-MF')].index
        # Initialize an empty DataFrame to store the results of the current file
        result_df = pd.DataFrame()
        # Get all data from the same station at the same time
        for idx in coliform_indices:
            station_id = df.iloc[idx, 2]
            timestamp = df.iloc[idx, 4]
            # Traverse upwards
            start_idx = idx
            while start_idx > 0 and df.iloc[start_idx - 1, 2] == station_id and df.iloc[start_idx - 1, 4] == timestamp:
                start_idx -= 1
            # Traverse downwards
            end_idx = idx
            while end_idx < len(df) - 1 and df.iloc[end_idx + 1, 2] == station_id and df.iloc[end_idx + 1, 4] == timestamp:
                end_idx += 1
            # Retain data
            station_data = df.iloc[start_idx:end_idx + 1]
            result_df = pd.concat([result_df, station_data], ignore_index=True)
        print(f'{filename} processed.')
        return result_df
    return pd.DataFrame()

if __name__ == '__main__':
    # Use multiprocessing to process files, with a custom number of processes set to 4
    with Pool(processes=4) as pool:
        results = pool.map(process_file, range(2000, 2025))

    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)

    # Delete the first two columns
    combined_df = combined_df.iloc[:, 2:]

    # Save the combined data to the 2000-2025Coliform.csv file
    combined_df.to_csv('2000-2025Coliform.csv', index=False)
    print('All files processed and combined data saved to 2000-2025Coliform.csv.')