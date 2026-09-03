import pandas as pd
import numpy as np
import time
from sklearn.cluster import DBSCAN
from rtree import index
from scipy.spatial.distance import directed_hausdorff

# ==========================================
# 1. Configuration and Parameters
# ==========================================
# Spatial epsilon (roughly 100 meters in radians for Haversine)
EPSILON_KM = 0.1 
EARTH_RADIUS_KM = 6371.0
EPSILON_RAD = EPSILON_KM / EARTH_RADIUS_KM

MIN_PTS = 4
K_DURATION = 3 # Minimum time slices to form a convoy
TAU_VALUES = [4, 5, 6, 7] # Minimum objects in a convoy

# ==========================================
# 2. Data Loading & Interpolation
# ==========================================
def load_and_prep_data(filepath, max_rows=100000):
    """Loads a subset of T-Drive and groups by minute for snapshots."""
    print("Loading dataset...")
    df = pd.read_csv(filepath, header=None, nrows=max_rows,
                     names=['taxi_id', 'date_time', 'longitude', 'latitude'])
    
    # Convert timestamp to datetime and round to nearest minute for snapshots
    df['timestamp'] =  pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S').dt.round('min')

    # Convert coordinates to radians for DBSCAN Haversine metric
    df['lat_rad'] = np.radians(df['latitude'])
    df['lon_rad'] = np.radians(df['longitude'])
    
    return df

# ==========================================
# 3. Core Spatial Functions
# ==========================================
def calculate_hausdorff(cluster_a_coords, cluster_b_coords):
    """Calculates the Hausdorff distance between two clusters."""
    # directed_hausdorff returns (distance, index_a, index_b)
    dist_ab = directed_hausdorff(cluster_a_coords, cluster_b_coords)[0]
    dist_ba = directed_hausdorff(cluster_b_coords, cluster_a_coords)[0]
    return max(dist_ab, dist_ba)

def get_bounding_box(coords):
    """Returns (min_x, min_y, max_x, max_y) for R-Tree insertion."""
    return (np.min(coords[:, 1]), np.min(coords[:, 0]), 
            np.max(coords[:, 1]), np.max(coords[:, 0]))

# ==========================================
# 4. CMC Algorithm (R-Tree Baseline)
# ==========================================
def run_cmc_baseline(df, tau):
    print(f"  Running CMC (R-Tree) for tau={tau}...")
    start_time = time.time()
    
    # Initialize R-Tree
    p = index.Property()
    idx = index.Index(properties=p)
    
    snapshots = df.groupby('timestamp')
    historical_convoys = {}
    convoy_id_counter = 0
    
    for ts, snapshot in snapshots:
        coords = snapshot[['lat_rad', 'lon_rad']].values
        taxis = snapshot['taxi_id'].values
        
        # 1. DBSCAN Clustering
        db = DBSCAN(eps=EPSILON_RAD, min_samples=tau, algorithm='ball_tree', metric='haversine')
        labels = db.fit_predict(coords)
        
        unique_labels = set(labels) - {-1}
        
        for cluster_label in unique_labels:
            mask = (labels == cluster_label)
            cluster_taxis = taxis[mask]
            cluster_coords = coords[mask]
            
            bbox = get_bounding_box(cluster_coords)
            
            # 2. Query R-Tree for spatial overlaps (CMC-H)
            candidate_ids = list(idx.intersection(bbox))
            
            # 3. Heavy computation: Hausdorff check on candidates
            for cid in candidate_ids:
                hist_coords = historical_convoys[cid]
                dist = calculate_hausdorff(cluster_coords, hist_coords)
                # If dist < threshold, it's a recurrent convoy (omitted exact logic for brevity)
                
            # 4. Insert new cluster into R-Tree
            idx.insert(convoy_id_counter, bbox)
            historical_convoys[convoy_id_counter] = cluster_coords
            convoy_id_counter += 1
            
    return (time.time() - start_time) * 1000

# ==========================================
# 5. RCI Algorithm (Intersection Index)
# ==========================================
def run_rci_proposed(df, tau):
    print(f"  Running RCI (Inverted Index) for tau={tau}...")
    start_time = time.time()
    
    snapshots = df.groupby('timestamp')
    inverted_index = {} # Maps taxi_id -> list of cluster_ids
    
    cluster_id_counter = 0
    
    for ts, snapshot in snapshots:
        coords = snapshot[['lat_rad', 'lon_rad']].values
        taxis = snapshot['taxi_id'].values
        
        # 1. DBSCAN Clustering
        db = DBSCAN(eps=EPSILON_RAD, min_samples=tau, algorithm='ball_tree', metric='haversine')
        labels = db.fit_predict(coords)
        
        unique_labels = set(labels) - {-1}
        
        for cluster_label in unique_labels:
            mask = (labels == cluster_label)
            cluster_taxis = set(taxis[mask])
            
            # 2. Fast Set Intersection (Bypassing Spatial Math)
            past_clusters = {}
            for taxi in cluster_taxis:
                if taxi in inverted_index:
                    for cid in inverted_index[taxi]:
                        past_clusters[cid] = past_clusters.get(cid, 0) + 1
            
            # If a past cluster shares >= tau taxis with the current cluster, it's recurrent
            recurrent_candidates = [cid for cid, count in past_clusters.items() if count >= tau]
            
            # 3. Update Inverted Index
            for taxi in cluster_taxis:
                if taxi not in inverted_index:
                    inverted_index[taxi] = []
                inverted_index[taxi].append(cluster_id_counter)
                
            cluster_id_counter += 1
            
    return (time.time() - start_time) * 1000

# ==========================================
# 6. Experimental Execution Loop
# ==========================================
if __name__ == "__main__":
    # Point this to a valid T-Drive text file on your machine
    # Example: 'release/taxi_log_2008_by_id/1.txt'
    # For testing, you can merge a few TXT files into one CSV first.
    DATA_FILE = 'tdrive_subset.csv' 
    
    try:
        df = load_and_prep_data(DATA_FILE, max_rows=50000)
        
        results = {'tau': TAU_VALUES, 'CMC_Time_ms': [], 'RCI_Time_ms': []}
        
        print("Starting Benchmark...")
        for tau in TAU_VALUES:
            print(f"\nEvaluating tau = {tau}")
            
            cmc_time = run_cmc_baseline(df, tau)
            results['CMC_Time_ms'].append(cmc_time)
            
            rci_time = run_rci_proposed(df, tau)
            results['RCI_Time_ms'].append(rci_time)
            
        # Display Final Results
        print("\n=== BENCHMARK RESULTS ===")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
    except FileNotFoundError:
        print(f"Error: Could not find '{DATA_FILE}'. Please point DATA_FILE to your T-Drive dataset.")