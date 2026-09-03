import pandas as pd
import numpy as np
import time as time
from sklearn.cluster import DBSCAN
import visualize_convoy_tdrive as viz

total_obj = 3
total_time = 3
total_dist = 100

def discover_convoys_cmc(df, m, k, e_meters):
    """
    Implementation of the Coherent Moving Cluster (CMC) algorithm 
    using Haversine distance for physical meters.
    """
    # Earth's radius in meters for Haversine conversion
    earth_radius_m = 6371000.0
    e_radians = e_meters / earth_radius_m

    # Ensure data is sorted by time in ascending order
    times = sorted(df['time'].unique())
    
    V = [] # Active convoy candidates
    V_result = [] # Final discovered convoys
    
    for t in times:
        V_next = []
        
        # Extract objects active at the current time point t
        df_t = df[df['time'] == t]
        
        # If there are fewer than 'm' objects total, no cluster can form
        if len(df_t) < m:
            for v in V:
                if v['lifetime'] >= k:
                    V_result.append(v)
            V = []
            continue
            
        # Haversine requires coordinates in [latitude, longitude] order (y, x)
        # and they must be converted to radians
        coords_rad = np.radians(df_t[['y', 'x']].values)
        obj_ids = df_t['id'].values
        
        # Snapshot clustering using DBSCAN with Haversine metric
        db = DBSCAN(eps=e_radians, min_samples=m, metric='haversine', algorithm='ball_tree').fit(coords_rad)
        labels = db.labels_
        
        # Group objects into snapshot clusters (ignore noise labeled as -1)
        C = []
        for label in set(labels):
            if label != -1:
                cluster_objects = set(obj_ids[labels == label])
                C.append({'objects': cluster_objects, 'assigned': False})
                
        # Compare current snapshot clusters with existing convoy candidates
        for v in V:
            v['assigned'] = False
            for c in C:
                # Check for common objects
                intersect = c['objects'].intersection(v['objects'])
                
                if len(intersect) >= m:
                    v['assigned'] = True
                    c['assigned'] = True
                    
                    # Update candidate with the intersection of objects
                    new_v = {
                        'objects': intersect,
                        'start_time': v['start_time'],
                        'end_time': t,
                        'lifetime': v['lifetime'] + 1,
                        'assigned': False
                    }
                    V_next.append(new_v)
                    
        # Check if any unassigned candidates have met the lifetime threshold 'k'
        for v in V:
            if not v['assigned'] and v['lifetime'] >= k:
                V_result.append(v)
                
        # Treat unassigned snapshot clusters as new convoy candidates
        for c in C:
            if not c['assigned']:
                new_c = {
                    'objects': c['objects'],
                    'start_time': t,
                    'end_time': t,
                    'lifetime': 1,
                    'assigned': False
                }
                V_next.append(new_c)
                
        # Move to the next time step
        V = V_next
        
    # Final check for any active candidates at the end of the time domain
    for v in V:
        if v['lifetime'] >= k:
            V_result.append(v)
            
    return V_result

# ==========================================
# Example Usage with T-Drive Data
# ==========================================
if __name__ == "__main__":
    
    file = 'tdrive_processed.csv'
    
    # Load the WHOLE file (do not use nrows=10000)
    print("Loading data...")
    df_temp = pd.read_csv(file, header=0, parse_dates=['time'])
    
    # --- NEW: FILTER BY A SPECIFIC TIME WINDOW ---
    # For demonstration, we will focus on a single day window (Feb 2, 2008, 13:00:00 to 18:00:00) to reduce the dataset size.
    start_time = pd.to_datetime("2008-02-02 13:00:00")
    end_time = pd.to_datetime("2008-02-02 18:00:00")
    
    df_temp = df_temp[(df_temp['time'] >= start_time) & (df_temp['time'] <= end_time)].copy()
    print(f"Total rows in one day window: {len(df_temp)}")
    print(f"Total unique taxis in window: {df_temp['id'].nunique()}")
    
    
    # Run the CMC Algorithm
    print("Running CMC Algorithm...")
    time_start = time.time()
    convoys = discover_convoys_cmc(df_temp, m=total_obj, k=total_time, e_meters=total_dist)
    time_end = time.time()
    print(f"Execution Time: {time_end - time_start:.2f} seconds")
    df_convoys = pd.DataFrame(convoys)
    df_convoys.to_csv(f"discovered_convoys_{total_obj}_{total_time}_{total_dist}.csv", index=False)
    
    
    print(f"\nTotal Convoys Discovered: {len(convoys)}")
    for i, c in enumerate(convoys): # Print first 5 to avoid console spam
        print(f"Convoy {i+1}: Objects {c['objects']} | Time: {c['start_time']} to {c['end_time']}")
        
    if convoys:
        viz.visualize_convoys(df_temp, convoys, file=f"all_convoys_map_beijing_{total_obj}_{total_time}_{total_dist}")  # Visualize all convoys