import pandas as pd
import os
from multiprocessing import Pool

def process_file(year):
    filename = 'D:\\Downloads\\' + f'{year}.csv'
    if os.path.exists(filename):
        print(f'Processing {filename}...')
        try:
            # Read the CSV file
            df = pd.read_csv(filename)
            # Check if the DataFrame has enough columns
            if df.shape[1] < 6:
                print(f'{filename} does not have enough columns.')
                return pd.DataFrame()
            # Initialize a list to store the results of the current file
            result_list = []
            # Get unique station_id and timestamp combinations with their start and end indices
            unique_combinations = df.drop_duplicates(subset=[df.columns[2], df.columns[4]], keep='first').index
            unique_combinations = list(unique_combinations) + [len(df)]  # Add the end of the DataFrame as the last index
            for i in range(len(unique_combinations) - 1):
                start_idx = unique_combinations[i]
                end_idx = unique_combinations[i + 1]
                station_data = df.iloc[start_idx:end_idx]
                # Check if any row in the filtered data contains coli data
                if station_data.iloc[:, 5].isin([
                    "Tot Coli CMF", "E.coli Pres", "E.coli Conf", "E.coli C-MF", "Colfm C-MF", "ColfmF C-MF", 
                    "Colfm P-MPN", "E.coli PMF", "Colfm PMF", "Colfm PMF10", "E.coli PMF10", "E.coli C-MPN", 
                    "Ecoli P-MPNB", "ColfmF Conf", "Colfrm Conf", "Colfm Conf", "ColfmF MF10", "ColformsPre", 
                    "F Coli Pre", "EColi HH2", "ColfmF PMF"
                ]).any():
                    result_list.append(station_data)
            result_df = pd.concat(result_list, ignore_index=True)
            print(f'{filename} processed.')
            return result_df
        except Exception as e:
            print(f'Error processing {filename}: {e}')
            return pd.DataFrame()
    return pd.DataFrame()

if __name__ == '__main__':
    # Use multiprocessing to process files, with a custom number of processes set to 4
    with Pool(processes=5) as pool:
        results = pool.map(process_file, range(2000, 2026))

    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)

    # Delete the first two columns
    combined_df = combined_df.iloc[:, 2:]

    # Save the combined data to the 2000-2025Coliform.csv file
    combined_df.to_csv('1 2000-2025Coliform.csv', index=False)
    print('All files processed and combined data saved to 2000-2025Coliform.csv.')