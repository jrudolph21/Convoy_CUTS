import time as time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import visualize_convoy_tdrive as viz
import cmc_example as cmc # Ensure this is the correct import path for your CMC implementation


# --- 1. Trajectory Simplification Helpers ---

def perpendicular_distance(point, start, end):
    """Calculates the perpendicular distance from a point to a line segment."""
    if np.all(start == end):
        return np.linalg.norm(point - start)
    
    line_vec = end - start
    point_vec = point - start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    
    # Project point onto the line
    proj = np.dot(point_vec, line_unitvec)
    
    # Clamp projection to the segment bounds
    if proj <= 0:
        return np.linalg.norm(point - start)
    elif proj >= line_len:
        return np.linalg.norm(point - end)
    else:
        proj_point = start + proj * line_unitvec
        return np.linalg.norm(point - proj_point)

def douglas_peucker_with_tolerance(points, times, delta):
    """
    Simplifies a trajectory using the Douglas-Peucker algorithm and 
    calculates the actual tolerance (max deviation) for each segment.
    """
    # FIX: Explicitly format the base case to guarantee a 2-point segment and 2 timestamps
    if len(points) < 3:
        return [np.array([points[0], points[-1]])], [(times[0], times[-1])], [0.0]
    
    start, end = points[0], points[-1]
    max_dist = 0.0
    index = 0
    
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i
            
    if max_dist > delta:
        # Recursive call if max distance exceeds the global tolerance delta
        left_points, left_times, left_tols = douglas_peucker_with_tolerance(points[:index+1], times[:index+1], delta)
        right_points, right_times, right_tols = douglas_peucker_with_tolerance(points[index:], times[index:], delta)
        
        # Combine results
        return (left_points + right_points, 
                left_times + right_times, 
                left_tols + right_tols)
    else:
        # Segment is valid; return the endpoints and the actual tolerance (max_dist)
        return [np.array([start, end])], [(times[0], times[-1])], [max_dist]


# --- 2. Filter Step (Algorithm 2) ---

def line_segment_distance(seg1, seg2):
    """
    Approximates the shortest Euclidean distance between two line segments.
    For production use, this can be optimized using bounding box pruning (Lemma 2).
    """
    # Simplified approximation: compute distance between endpoints
    # A robust implementation would calculate exact segment-to-segment distance
    dists = [
        np.linalg.norm(seg1[0] - seg2[0]),
        np.linalg.norm(seg1[0] - seg2[1]),
        np.linalg.norm(seg1[1] - seg2[0]),
        np.linalg.norm(seg1[1] - seg2[1])
    ]
    return min(dists)

def traj_dbscan(segments, eps, min_samples):
    """
    Optimized Trajectory DBSCAN using Bounding Box (MBR) Pruning 
    and NumPy Vectorization.
    """
    n = len(segments)
    if n == 0:
        return []

    # Initialize matrix with penalty distance (eps + 10.0)
    dist_matrix = np.full((n, n), eps + 10.0)
    np.fill_diagonal(dist_matrix, 0.0)

    # Vectorize segment arrays: P1 (starts), P2 (ends), and tolerances
    P1 = np.array([s[0][0] for s in segments])  # Shape: (N, 2)
    P2 = np.array([s[0][1] for s in segments])  # Shape: (N, 2)
    tols = np.array([s[1] for s in segments])   # Shape: (N,)

    # Precompute Minimum Bounding Rectangles (MBRs) for all segments
    min_x = np.minimum(P1[:, 0], P2[:, 0])
    max_x = np.maximum(P1[:, 0], P2[:, 0])
    min_y = np.minimum(P1[:, 1], P2[:, 1])
    max_y = np.maximum(P1[:, 1], P2[:, 1])

    deg_conv = 111000.0  # Approximation: ~111,000 meters per degree

    for i in range(n - 1):
        # Convert dynamic CuTS epsilon tolerance to degrees
        adj_eps_deg = (eps + tols[i] + tols[i + 1:]) / deg_conv

        # 1. BOUNDING BOX PRUNING (Fast Spatial Filter)
        # Skip distance calculation entirely if bounding boxes do not overlap
        overlap_mask = ~(
            (max_x[i] + adj_eps_deg < min_x[i + 1:]) |
            (min_x[i] - adj_eps_deg > max_x[i + 1:]) |
            (max_y[i] + adj_eps_deg < min_y[i + 1:]) |
            (min_y[i] - adj_eps_deg > max_y[i + 1:])
        )

        if not np.any(overlap_mask):
            continue

        # Indices of candidates that passed the spatial filter
        cand_indices = np.where(overlap_mask)[0] + (i + 1)

        # 2. VECTORIZED DISTANCE CALCULATION
        A, B = P1[i], P2[i]
        C, D = P1[cand_indices], P2[cand_indices]

        # Batch endpoint distances
        d_AC = np.linalg.norm(A - C, axis=1)
        d_AD = np.linalg.norm(A - D, axis=1)
        d_BC = np.linalg.norm(B - C, axis=1)
        d_BD = np.linalg.norm(B - D, axis=1)

        # Convert minimum endpoint distance back to meters
        min_dists_m = np.minimum(np.minimum(d_AC, d_AD), np.minimum(d_BC, d_BD)) * deg_conv

        # Check against dynamic threshold (Lemma 1)
        req_eps_m = eps + tols[i] + tols[cand_indices]
        valid_neighbors = cand_indices[min_dists_m <= req_eps_m]

        # Set valid neighbor distances to 0.0 for DBSCAN
        dist_matrix[i, valid_neighbors] = 0.0
        dist_matrix[valid_neighbors, i] = 0.0

    # Fit DBSCAN using C-optimized precomputed matrix
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist_matrix)

    clusters = []
    for label in set(db.labels_):
        if label != -1:
            cluster_objects = {segments[idx][2] for idx in np.where(db.labels_ == label)[0]}
            clusters.append({'objects': cluster_objects, 'assigned': False})

    return clusters

def cuts_filter(df, m, k, e_meters, delta, lambda_time):
    """
    Algorithm 2: CuTS Filter Step
    """
    simplified_trajectories = {}
    
    # Convert 100 meters to degrees (~111,000 meters per degree)
    delta_deg = delta / 111000.0 
    # Then pass delta_deg to douglas_peucker_with_tolerance

    # 1. Simplify all trajectories
    for obj_id, group in df.groupby('id'):
        
        # FIX: Skip objects that only have a single point
        if len(group) < 2:
            continue
            
        points = group[['x', 'y']].values
        times = group['time'].values
        segs, time_spans, tols = douglas_peucker_with_tolerance(points, times, delta_deg)
        tols_meters = [t * 111000.0 for t in tols]
        simplified_trajectories[obj_id] = list(zip(segs, time_spans, tols_meters))
        
    # 2. Divide the time domain into disjoint partitions of length lambda_time
    min_time = df['time'].min()
    max_time = df['time'].max()
    
    V = []
    V_cand = []
    
    current_time = min_time
    while current_time <= max_time:
        next_time = current_time + pd.Timedelta(minutes=lambda_time)
        V_next = []
        
        # Collect line segments overlapping this time partition Tz
        G = [] 
        for obj_id, traj in simplified_trajectories.items():
            for seg, (t_start, t_end), tol in traj:
                if t_start <= next_time and t_end >= current_time:
                    G.append((seg, tol, obj_id))
                    
        # Apply Trajectory DBSCAN
        C = traj_dbscan(G, e_meters, m)
        
        # Intersect with previous candidates
        for v in V:
            v['assigned'] = False
            for c in C:
                intersect = c['objects'].intersection(v['objects'])
                if len(intersect) >= m:
                    v['assigned'] = True
                    c['assigned'] = True
                    new_v = {
                        'objects': intersect,
                        'start_time': v['start_time'],
                        'end_time': next_time,
                        'lifetime': v['lifetime'] + lambda_time,
                        'assigned': False
                    }
                    V_next.append(new_v)
                    
            if not v['assigned'] and (v['lifetime'] / lambda_time) >= k:
                V_cand.append(v)
                
        for c in C:
            if not c['assigned']:
                V_next.append({
                    'objects': c['objects'],
                    'start_time': current_time,
                    'end_time': next_time,
                    'lifetime': lambda_time,
                    'assigned': False
                })
                
        V = V_next
        current_time = next_time
        
    for v in V:
        if (v['lifetime'] / lambda_time) >= k:
            V_cand.append(v)
            
    return V_cand

# --- 3. Refinement Step (Algorithm 3) ---

def cuts_refinement(df, candidates, m, k_cmc, e_meters):
    """
    Algorithm 3: CuTS Refinement Step
    Filters the original dataset based on the candidate time bounds and objects,
    then runs the rigorous CMC algorithm to extract exact convoys.
    """
    final_convoys = []
    
    for v in candidates:
        t_start = v['start_time']
        t_end = v['end_time']
        candidate_objects = v['objects']
        
        # Extract subset of original trajectories for this candidate's timeframe
        df_subset = df[
            (df['id'].isin(candidate_objects)) & 
            (df['time'] >= t_start) & 
            (df['time'] <= t_end)
        ]
        
        # Run your existing CMC algorithm on the refined subset
        # Ensure 'discover_convoys_cmc' is defined and loaded from your code
        refined_convoys = cmc.discover_convoys_cmc(df_subset, m, k_cmc, e_meters)
        final_convoys.extend(refined_convoys)
        
    return final_convoys

def discover_convoys_cuts(df, m, k_cuts, k_cmc, e_meters, delta, lambda_time):
    """Main wrapper function executing and timing Filter and Refinement steps."""
    
    total_start = time.time()
    
    # 1. TIME THE FILTER STEP
    print("Running Original CuTS Filter Step...")
    filter_start = time.time()
    candidates = cuts_filter(df, m, k_cuts, e_meters, delta, lambda_time)
    filter_end = time.time()
    filter_duration = filter_end - filter_start
    
    print(f"Filter complete in {filter_duration:.2f} seconds. Found {len(candidates)} candidate convoys.")
    
    # 2. TIME THE REFINEMENT STEP
    print("Running Refinement Step...")
    refinement_start = time.time()
    actual_convoys = cuts_refinement(df, candidates, m, k_cmc, e_meters)
    refinement_end = time.time()
    refinement_duration = refinement_end - refinement_start
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    # PRINT TIMING BREAKDOWN
    print("\n" + "="*40)
    print("       CUTS TIMING BREAKDOWN          ")
    print("="*40)
    print(f"Filter Time     : {filter_duration:.2f} seconds ({filter_duration/total_duration*100:.1f}%)")
    print(f"Refinement Time : {refinement_duration:.2f} seconds ({refinement_duration/total_duration*100:.1f}%)")
    print(f"Total Time      : {total_duration:.2f} seconds")
    print("="*40 + "\n")
    
    return actual_convoys

if __name__ == "__main__":

    file = 'tdrive_processed.csv'

    # ==========================================================
    # Load EXACTLY the same dataset configuration as cmc_example
    # ==========================================================

    print("Loading data...")

    my_data = pd.read_csv(
        file,
        header=0,
        parse_dates=['time']
    )

    print(f"Data loaded. Total rows: {len(my_data)}")
    print(f"Time dtype: {my_data['time'].dtype}")

    # ==========================================================
    # EXACT SAME TIME WINDOW AS CMC
    # ==========================================================

    start_time = pd.Timestamp("2008-02-02 13:00:00")
    end_time = pd.Timestamp("2008-02-02 18:00:00")

    windowed_data = my_data[
        (my_data['time'] >= start_time) &
        (my_data['time'] <= end_time)
    ].copy()

    print(f"\nData filtered from {start_time} to {end_time}")
    print(f"Original data points: {len(my_data)}")
    print(f"Windowed data points: {len(windowed_data)}")
    print(f"Unique taxis: {windowed_data['id'].nunique()}")

    # ==========================================================
    # VERIFY DATA IS IDENTICAL TO CMC INPUT
    # ==========================================================

    print("\nDataset verification:")
    print(f"Rows: {len(windowed_data)}")
    print(f"Columns: {list(windowed_data.columns)}")
    print(f"Time dtype: {windowed_data['time'].dtype}")
    print(f"First timestamp: {windowed_data['time'].min()}")
    print(f"Last timestamp: {windowed_data['time'].max()}")

    # ==========================================================
    # SAME CMC PARAMETERS
    # ==========================================================

    m_objects = 3
    k_cuts = 2
    k_cmc = 3
    epsilon = 100.0

    # ==========================================================
    # CuTS-SPECIFIC PARAMETERS
    # ==========================================================

    simplification_tolerance = 100.0
    time_window = 10

    # ==========================================================
    # RUN CuTS
    # ==========================================================

    print("\nRunning CuTS...")

    convoys = discover_convoys_cuts(
        df=windowed_data,
        m=m_objects,
        k_cuts=k_cuts,
        k_cmc=k_cmc,
        e_meters=epsilon,
        delta=simplification_tolerance,
        lambda_time=time_window
    )

    print(f"\nTotal CuTS Convoys Discovered: {len(convoys)}")

    for i, c in enumerate(convoys):
        print(
            f"Convoy {i+1}: "
            f"Objects {c['objects']} | "
            f"Time: {c['start_time']} to {c['end_time']}"
        )

    # ==========================================================
    # VISUALIZE
    # Use the SAME windowed dataset that CuTS processed
    # ==========================================================

    if convoys:
        viz.visualize_convoys(
            windowed_data,
            convoys,
            file="cuts_convoys_map"
        )