import pandas as pd
import numpy as np
import glob
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in meters.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000.0 # Radius of earth in meters
    return c * r

def load_and_preprocess_tdrive(input_directory, output_file):
    print(f"Locating files in {input_directory}...")
    
    # Grab ALL files 
    all_files = glob.glob(f"{input_directory}/*.txt")
    print(f"Found {len(all_files)} files. Reading data...")
    
    df_list = []
    for file in all_files:
        # Map the column names directly to the 'id', 'time', 'x', 'y' format used by the pipeline
        df_temp = pd.read_csv(file, header=None, names=['id', 'time', 'x', 'y'])
        df_list.append(df_temp)
        
    # Concatenate all files into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)
    
    initial_rows = len(df)
    print(f"\nInitial row count: {initial_rows}")
    
    # 1. Clean invalid coordinates (drops 0,0 anomalies)
    print("--> Removing invalid coordinates...")
    df = df[(df['x'] > 100) & (df['y'] > 30)].copy()
    
    # 2. Convert to datetime
    print("--> Converting timestamps to datetime objects...")
    df['time'] = pd.to_datetime(df['time'])
    
    # 3. Sort chronologically per taxi
    print("--> Sorting data by taxi ID and time...")
    df = df.sort_values(by=['id', 'time']).reset_index(drop=True)
    
    # 5. Bin timestamps to the nearest 10 minutes
    print("--> Binning timestamps to the nearest 10 minutes...")
    df['time'] = df['time'].dt.round('10min')
    
    # 6. Deduplicate (keep only one ping per taxi per 10-minute bin)
    print("--> Removing duplicate time bins per taxi...")
    df = df.drop_duplicates(subset=['id', 'time'], keep='first').reset_index(drop=True)
    
    final_rows = len(df)
    print(f"\nFinal row count: {final_rows} (Removed {initial_rows - final_rows} rows during cleaning)")
    
    # 7. Save to CSV
    print(f"Saving preprocessed data to {output_file}...")
    df.to_csv(output_file, index=False) 
    print("Preprocessing complete!")

if __name__ == "__main__":
    # Directory containing the original .txt files
    INPUT_DIRECTORY = "release/taxi_log_2008_by_id"
    
    # File to save the final cleaned dataset to
    OUTPUT_FILE = "tdrive_processed_nospd.csv"
    
    # Run the pipeline
    load_and_preprocess_tdrive(INPUT_DIRECTORY, OUTPUT_FILE)