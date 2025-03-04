import os
import requests
from concurrent.futures import ThreadPoolExecutor

def download_csv(year, download_dir):
    url = f"https://environment.data.gov.uk/water-quality/batch/result-download/{year}.csv"
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(download_dir, f"{year}.csv")
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {year}.csv")
    else:
        print(f"Failed to download {year}.csv")

def main():
    download_dir = "D:/Downloads"
    os.makedirs(download_dir, exist_ok=True)
    years = range(2000, 2026)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda year: download_csv(year, download_dir), years)

if __name__ == "__main__":
    main()
