"""
This script fetches the COVID-19 dataset from Our World in Data.
It provides functions to download the latest COVID-19 data and save it locally.
"""

import os
import requests
import pandas as pd
from datetime import datetime

# Define constants
DATA_DIR = "data"
OWID_COVID_DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
DATA_FILE_PATH = os.path.join(DATA_DIR, "owid-covid-data.csv")
JHU_GITHUB_URL = "https://github.com/CSSEGISandData/COVID-19"
TIMESTAMP_FILE = os.path.join(DATA_DIR, "last_updated.txt")

def create_data_directory():
    """Create data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def download_owid_data():
    """Download the latest COVID-19 data from Our World in Data."""
    create_data_directory()
    
    try:
        print(f"Downloading COVID-19 data from Our World in Data...")
        response = requests.get(OWID_COVID_DATA_URL)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        with open(DATA_FILE_PATH, 'wb') as file:
            file.write(response.content)
        
        # Save timestamp
        with open(TIMESTAMP_FILE, 'w') as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        print(f"Data successfully downloaded and saved to {DATA_FILE_PATH}")
        
        # Verify data by loading it with pandas
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Downloaded data contains {df.shape[0]} rows and {df.shape[1]} columns.")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return True
    
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

def load_covid_data():
    """Load COVID-19 data from local file or download if not available."""
    if not os.path.exists(DATA_FILE_PATH):
        print("COVID-19 data file not found locally. Downloading...")
        download_owid_data()
    
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Loaded COVID-19 data with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_last_updated_time():
    """Get the timestamp of when the data was last updated."""
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, 'r') as f:
            return f.read().strip()
    return "Unknown"

def data_status():
    """Check data status and provide information."""
    if os.path.exists(DATA_FILE_PATH):
        file_size_mb = os.path.getsize(DATA_FILE_PATH) / (1024 * 1024)
        last_modified = datetime.fromtimestamp(os.path.getmtime(DATA_FILE_PATH)).strftime("%Y-%m-%d %H:%M:%S")
        last_updated = get_last_updated_time()
        
        print(f"COVID-19 Data Status:")
        print(f"- File: {DATA_FILE_PATH}")
        print(f"- Size: {file_size_mb:.2f} MB")
        print(f"- Last Modified: {last_modified}")
        print(f"- Last Updated: {last_updated}")
        
        # Load data to get date range
        try:
            df = pd.read_csv(DATA_FILE_PATH)
            print(f"- Data Range: {df['date'].min()} to {df['date'].max()}")
            print(f"- Countries/Regions: {df['location'].nunique()}")
        except Exception as e:
            print(f"- Error reading data: {e}")
    else:
        print("COVID-19 data file not found locally.")

if __name__ == "__main__":
    print("COVID-19 Data Collection Script")
    print("=" * 40)
    
    # Check if data exists
    if os.path.exists(DATA_FILE_PATH):
        print(f"COVID-19 data file already exists at {DATA_FILE_PATH}")
        print("Do you want to update the data? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            download_owid_data()
    else:
        download_owid_data()
    
    # Show data status
    data_status()
