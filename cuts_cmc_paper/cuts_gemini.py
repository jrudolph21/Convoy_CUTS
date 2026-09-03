import time as time
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import visualize_convoy_tdrive as viz
import cmc_example as cmc  # Ensure this is the correct import path for your CMC implementation


# --- 1. Trajectory Simplification Helpers ---

def perpendicular_distance(point, start, end):
    """Calculates perpendicular distance from a point to a line segment in degrees."""
    if np.all(start == end):
        return np.linalg.norm(point - start)
    
    line_vec = end - start
    point_vec = point - start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    
    proj = np.dot(point_vec, line_unitvec)
    
    if proj <= 0:
        return np.linalg.norm(point - start)
    elif proj >= line_len:
        return np.linalg.norm(point - end)
    else:
        proj_point = start + proj * line_unitvec
        return np.linalg.norm(point - proj_point)

def douglas_peucker_with_tolerance(points, times, delta):
    """
    Simplifies a trajectory using Douglas-Peucker algorithm.
    Preserves all segment lists without truncation.
    """
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
        left_points, left_times, left_tols = douglas_peucker_with_tolerance(points[:index+1], times[:index+1], delta)
        right_points, right_times, right_tols = douglas_peucker_with_tolerance(points[index:], times[index:], delta)
        
        return (left_points + right_points, 
                left_times + right_times, 
                left_tols + right_tols)
    else:
        return [np.array([start, end])], [(times[0], times[-1])], [max_dist]


# --- 2. Geometric Segment-to-Segment Distance Helpers (Vectorized in Meters) ---

def dist_point_to_segments(P, C, D):
    """Calculates perpendicular distance in meters from a single point P to M segments (C, D)."""
    CD = D - C  # (M, 2)
    len_sq = np.sum(CD**2, axis=1)  # (M,)
    
    CP = P - C  # (M, 2)
    dot_prod = np.sum(CP * CD, axis=1)  # (M,)
    
    t = np.zeros_like(dot_prod)
    valid = len_sq > 0
    t[valid] = dot_prod[valid] / len_sq[valid]
    t = np.clip(t, 0.0, 1.0)
    
    proj = C + t[:, np.newaxis] * CD  # (M, 2)
    return np.linalg.norm(P - proj, axis=1)

def dist_points_to_segment(P, A, B):
    """Calculates perpendicular distance in meters from M points P to a single segment (A, B)."""
    AB = B - A  # (2,)
    len_sq = np.dot(AB, AB)
    if len_sq == 0:
        return np.linalg.norm(P - A, axis=1)
    
    AP = P - A  # (M, 2)
    dot_prod = np.dot(AP, AB)  # (M,)
    t = np.clip(dot_prod / len_sq, 0.0, 1.0)
    proj = A + t[:, np.newaxis] * AB  # (M, 2)
    return np.linalg.norm(P - proj, axis=1)


# --- 3. Filter Step (Algorithm 2) ---

def traj_dbscan(segments, eps, min_samples):
    """
    Trajectory DBSCAN using meter conversion and true segment-to-segment distance.
    """
    n = len(segments)
    if n == 0:
        return []

    dist_matrix = np.full((n, n), eps + 10.0)
    np.fill_diagonal(dist_matrix, 0.0)

    # Coordinates in degrees: [lon, lat]
    P1_deg = np.array([s[0][0] for s in segments])  # (N, 2)
    P2_deg = np.array([s[0][1] for s in segments])  # (N, 2)
    tols = np.array([s[1] for s in segments])      # Tolerances in meters

    # Beijing coordinate projection to meters (~39.9° N latitude)
    kx = 111000.0 * np.cos(np.radians(39.9))  # ~85,160 m/deg
    ky = 111000.0                             # 111,000 m/deg
    scale = np.array([kx, ky])

    P1_m = P1_deg * scale
    P2_m = P2_deg * scale

    min_x = np.minimum(P1_m[:, 0], P2_m[:, 0])
    max_x = np.maximum(P1_m[:, 0], P2_m[:, 0])
    min_y = np.minimum(P1_m[:, 1], P2_m[:, 1])
    max_y = np.maximum(P1_m[:, 1], P2_m[:, 1])

    for i in range(n - 1):
        # 1. Bounding Box Pruning (in meters) calculated against ALL remaining segments
        req_eps_all = eps + tols[i] + tols[i + 1:]

        overlap_mask = ~(
            (max_x[i] + req_eps_all < min_x[i + 1:]) |
            (min_x[i] - req_eps_all > max_x[i + 1:]) |
            (max_y[i] + req_eps_all < min_y[i + 1:]) |
            (min_y[i] - req_eps_all > max_y[i + 1:])
        )

        if not np.any(overlap_mask):
            continue

        cand_indices = np.where(overlap_mask)[0] + (i + 1)

        # 2. True Segment-to-Segment Distance Calculation
        A_i, B_i = P1_m[i], P2_m[i]
        C, D = P1_m[cand_indices], P2_m[cand_indices]

        d_Ai_CD = dist_point_to_segments(A_i, C, D)
        d_Bi_CD = dist_point_to_segments(B_i, C, D)
        d_C_ABi = dist_points_to_segment(C, A_i, B_i)
        d_D_ABi = dist_points_to_segment(D, A_i, B_i)

        seg_dists_m = np.minimum(np.minimum(d_Ai_CD, d_Bi_CD), np.minimum(d_C_ABi, d_D_ABi))

        # FIX: Calculate required epsilon ONLY for the remaining valid candidates
        req_eps_cand = eps + tols[i] + tols[cand_indices]

        # Both seg_dists_m and req_eps_cand now have identical shapes (e.g., 291,)
        valid_neighbors = cand_indices[seg_dists_m <= req_eps_cand]

        dist_matrix[i, valid_neighbors] = 0.0
        dist_matrix[valid_neighbors, i] = 0.0

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist_matrix)

    clusters = []
    for label in set(db.labels_):
        if label != -1:
            cluster_objects = {segments[idx][2] for idx in np.where(db.labels_ == label)[0]}
            clusters.append({'objects': cluster_objects, 'assigned': False})

    return clusters

def cuts_filter(df, m, k, e_meters, delta, lambda_time):
    """Algorithm 2: CuTS Filter Step"""
    simplified_trajectories = {}
    
    delta_deg = delta / 111000.0 

    for obj_id, group in df.groupby('id'):
        if len(group) < 2:
            continue
            
        points = group[['x', 'y']].values
        times = group['time'].values
        segs, time_spans, tols = douglas_peucker_with_tolerance(points, times, delta_deg)
        tols_meters = [t * 111000.0 for t in tols]
        simplified_trajectories[obj_id] = list(zip(segs, time_spans, tols_meters))
        
    min_time = df['time'].min()
    max_time = df['time'].max()
    
    V = []
    V_cand = []
    
    current_time = min_time
    while current_time <= max_time:
        next_time = current_time + pd.Timedelta(minutes=lambda_time)
        V_next = []
        
        G = [] 
        for obj_id, traj in simplified_trajectories.items():
            for seg, (t_start, t_end), tol in traj:
                if t_start <= next_time and t_end >= current_time:
                    G.append((seg, tol, obj_id))
                    
        C = traj_dbscan(G, e_meters, m)
        
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


# --- 4. Refinement Step (Algorithm 3) ---

def cuts_refinement(df, candidates, m, k_cmc, e_meters, lambda_time=10):
    """
    Algorithm 3: CuTS Refinement Step with temporal buffer to prevent boundary clipping.
    """
    final_convoys = []
    buffer = pd.Timedelta(minutes=lambda_time)
    
    for v in candidates:
        t_start = v['start_time'] - buffer
        t_end = v['end_time'] + buffer
        candidate_objects = v['objects']
        
        df_subset = df[
            (df['id'].isin(candidate_objects)) & 
            (df['time'] >= t_start) & 
            (df['time'] <= t_end)
        ]
        
        refined_convoys = cmc.discover_convoys_cmc(df_subset, m, k_cmc, e_meters)
        final_convoys.extend(refined_convoys)
        
    # Deduplicate refined convoys
    unique_convoys = []
    seen = set()
    for c in final_convoys:
        key = (frozenset(c['objects']), c['start_time'], c['end_time'])
        if key not in seen:
            seen.add(key)
            unique_convoys.append(c)
            
    return unique_convoys

def discover_convoys_cuts(df, m, k_cuts, k_cmc, e_meters, delta, lambda_time):
    """Main wrapper function executing and timing Filter and Refinement steps."""
    
    total_start = time.time()
    
    print("Running Original CuTS Filter Step...")
    filter_start = time.time()
    candidates = cuts_filter(df, m, k_cuts, e_meters, delta, lambda_time)
    filter_end = time.time()
    filter_duration = filter_end - filter_start
    
    print(f"Filter complete in {filter_duration:.2f} seconds. Found {len(candidates)} candidate convoys.")
    
    print("Running Refinement Step...")
    refinement_start = time.time()
    actual_convoys = cuts_refinement(df, candidates, m, k_cmc, e_meters, lambda_time)
    refinement_end = time.time()
    refinement_duration = refinement_end - refinement_start
    
    total_end = time.time()
    total_duration = total_end - total_start
    
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

    print("Loading data...")

    my_data = pd.read_csv(
        file,
        header=0,
        parse_dates=['time']
    )

    print(f"Data loaded. Total rows: {len(my_data)}")

    start_time = pd.Timestamp("2008-02-02 13:00:00")
    end_time = pd.Timestamp("2008-02-02 18:00:00")

    windowed_data = my_data[
        (my_data['time'] >= start_time) &
        (my_data['time'] <= end_time)
    ].copy()

    print(f"\nData filtered from {start_time} to {end_time}")
    print(f"Windowed data points: {len(windowed_data)}")
    print(f"Unique taxis: {windowed_data['id'].nunique()}")

    m_objects = 3
    k_cuts = 2        # 2 intervals (20 mins elapsed)
    k_cmc = 3         # 3 discrete pings
    epsilon = 100.0

    simplification_tolerance = 100.0
    time_window = 10

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

    if convoys:
        viz.visualize_convoys(
            windowed_data,
            convoys,
            file="cuts_convoys_map"
        )