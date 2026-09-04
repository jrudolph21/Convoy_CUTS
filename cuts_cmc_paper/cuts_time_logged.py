"""
CuTS convoy discovery on a preprocessed T-drive dataset.

Based on:
    Jeung et al., "Discovery of Convoys in Trajectory Databases"
    VLDB 2008.

Pipeline:
    1. Select the same window as the CMC example: 2008-02-02 13:00-18:00.
    2. Douglas-Peucker trajectory simplification (CuTS filter).
    3. Partition the window into lambda-sized temporal partitions.
    4. Cluster simplified trajectories in each partition.
    5. Track candidate convoys across adjacent partitions.
    6. Refine candidates on the original 10-minute snapshots using DBSCAN.
    7. Save CSV results and visualize discovered convoys.

Expected preprocessed CSV columns:
    id,time,x,y
where x=longitude and y=latitude.

Install if needed:
    pip install pandas numpy scikit-learn
"""

import math
import sys
import time
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import visualize_convoy_tdrive as viz


class Tee:
    """Write the same stream output to both the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)


total_obj = 3
total_time = 18
total_dist = 40
dp_tolerance = 31.5
lambda_points = 4


# ============================================================
# Geometry
# ============================================================

def point_to_segment_distance(px: float, py: float,
                              ax: float, ay: float,
                              bx: float, by: float) -> float:
    """Euclidean distance from point P to segment AB."""
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def orientation(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def on_segment(ax, ay, bx, by, px, py) -> bool:
    return (
        min(ax, bx) - 1e-12 <= px <= max(ax, bx) + 1e-12
        and min(ay, by) - 1e-12 <= py <= max(ay, by) + 1e-12
    )


def segments_intersect(a: dict, b: dict) -> bool:
    o1 = orientation(a['x0'], a['y0'], a['x1'], a['y1'], b['x0'], b['y0'])
    o2 = orientation(a['x0'], a['y0'], a['x1'], a['y1'], b['x1'], b['y1'])
    o3 = orientation(b['x0'], b['y0'], b['x1'], b['y1'], a['x0'], a['y0'])
    o4 = orientation(b['x0'], b['y0'], b['x1'], b['y1'], a['x1'], a['y1'])

    if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and \
       ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)):
        return True

    if abs(o1) < 1e-12 and on_segment(a['x0'], a['y0'], a['x1'], a['y1'], b['x0'], b['y0']):
        return True
    if abs(o2) < 1e-12 and on_segment(a['x0'], a['y0'], a['x1'], a['y1'], b['x1'], b['y1']):
        return True
    if abs(o3) < 1e-12 and on_segment(b['x0'], b['y0'], b['x1'], b['y1'], a['x0'], a['y0']):
        return True
    if abs(o4) < 1e-12 and on_segment(b['x0'], b['y0'], b['x1'], b['y1'], a['x1'], a['y1']):
        return True
    return False


def segment_distance(a: dict, b: dict) -> float:
    """Minimum 2D Euclidean distance between two line segments."""
    if segments_intersect(a, b):
        return 0.0

    return min(
        point_to_segment_distance(a['x0'], a['y0'], b['x0'], b['y0'], b['x1'], b['y1']),
        point_to_segment_distance(a['x1'], a['y1'], b['x0'], b['y0'], b['x1'], b['y1']),
        point_to_segment_distance(b['x0'], b['y0'], a['x0'], a['y0'], a['x1'], a['y1']),
        point_to_segment_distance(b['x1'], b['y1'], a['x0'], a['y0'], a['x1'], a['y1']),
    )


# ============================================================
# Douglas-Peucker simplification with segment tolerances
# ============================================================

def _dp_recursive(points: np.ndarray, indices: np.ndarray, delta: float) -> List[int]:
    """Return retained indices for spatial Douglas-Peucker."""
    if len(points) <= 2:
        return [int(indices[0]), int(indices[-1])] if len(points) == 2 else [int(indices[0])]

    a = points[0]
    b = points[-1]

    dists = np.array([
        point_to_segment_distance(float(p[0]), float(p[1]),
                                  float(a[0]), float(a[1]),
                                  float(b[0]), float(b[1]))
        for p in points[1:-1]
    ])

    if len(dists) == 0:
        return [int(indices[0]), int(indices[-1])]

    local_max = int(np.argmax(dists)) + 1
    max_dist = float(dists[local_max - 1])

    if max_dist <= delta:
        return [int(indices[0]), int(indices[-1])]

    left = _dp_recursive(points[: local_max + 1], indices[: local_max + 1], delta)
    right = _dp_recursive(points[local_max:], indices[local_max:], delta)
    return left[:-1] + right


def douglas_peucker_with_tolerance(group: pd.DataFrame, delta: float) -> Tuple[pd.DataFrame, List[dict]]:
    """Simplify one trajectory and calculate each segment's actual max DP error."""
    g = group.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    if len(g) < 2:
        return g.copy(), []

    points = g[['x', 'y']].to_numpy(dtype=float)
    idx = np.arange(len(g), dtype=int)
    keep = _dp_recursive(points, idx, delta)
    keep = sorted(set(keep))

    simp = g.iloc[keep].copy().reset_index(drop=True)
    segments: List[dict] = []

    for a_idx, b_idx in zip(keep[:-1], keep[1:]):
        a = g.iloc[a_idx]
        b = g.iloc[b_idx]
        if a['time'] == b['time']:
            continue

        interval_points = g.iloc[a_idx : b_idx + 1]
        max_error = 0.0
        for _, p in interval_points.iterrows():
            d = point_to_segment_distance(
                float(p['x']), float(p['y']),
                float(a['x']), float(a['y']),
                float(b['x']), float(b['y'])
            )
            max_error = max(max_error, d)

        segments.append(
            dict(
                taxi_id=int(a['id']),
                t0=pd.Timestamp(a['time']),
                t1=pd.Timestamp(b['time']),
                x0=float(a['x']),
                y0=float(a['y']),
                x1=float(b['x']),
                y1=float(b['y']),
                tolerance=max_error,
            )
        )

    return simp, segments


# ============================================================
# CuTS trajectory-distance filter
# ============================================================

def segment_interval_overlap(a: dict, b: dict,
                              window_start: pd.Timestamp,
                              window_end: pd.Timestamp) -> bool:
    a0 = max(a['t0'], window_start)
    a1 = min(a['t1'], window_end)
    b0 = max(b['t0'], window_start)
    b1 = min(b['t1'], window_end)
    return max(a0, b0) <= min(a1, b1)


def trajectory_filter_distance(segments_a: Sequence[dict],
                               segments_b: Sequence[dict],
                               window_start: pd.Timestamp,
                               window_end: pd.Timestamp) -> float:
    """
    CuTS-style omega bound:
        min(D_LL(segment_a, segment_b) - tol_a - tol_b)
    over temporally overlapping segment pairs.
    """
    best = float('inf')
    for a in segments_a:
        if a['t1'] < window_start or a['t0'] > window_end:
            continue
        for b in segments_b:
            if b['t1'] < window_start or b['t0'] > window_end:
                continue
            if not segment_interval_overlap(a, b, window_start, window_end):
                continue
            d = max(0.0, segment_distance(a, b) - a['tolerance'] - b['tolerance'])
            best = min(best, d)
            if best <= 0.0:
                return 0.0
    return best


def build_trajectory_distance_matrix(
    partition_segments: Dict[int, List[dict]],
    epsilon: float,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> Tuple[List[int], np.ndarray]:
    """Build precomputed distances for simplified trajectories in one partition."""
    ids = sorted(partition_segments)
    n = len(ids)
    D = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(D, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            d = trajectory_filter_distance(
                partition_segments[ids[i]],
                partition_segments[ids[j]],
                window_start,
                window_end,
            )
            D[i, j] = D[j, i] = d

    return ids, D


def traj_dbscan(partition_segments: Dict[int, List[dict]],
                epsilon: float,
                min_objects: int,
                window_start: pd.Timestamp,
                window_end: pd.Timestamp) -> List[Set[int]]:
    """
    CuTS filter clustering over simplified trajectories.

    The paper calls this TRAJ-DBSCAN. We use DBSCAN with a precomputed
    trajectory-to-trajectory distance matrix built from the CuTS distance bound.
    """
    if len(partition_segments) < min_objects:
        return []

    ids, D = build_trajectory_distance_matrix(
        partition_segments, epsilon, window_start, window_end
    )

    # Values above epsilon cannot be neighbors. Keep them as finite values
    # because sklearn's precomputed DBSCAN requires a numeric matrix.
    D_db = D.copy()
    D_db[~np.isfinite(D_db)] = epsilon + 1e9

    model = DBSCAN(
        eps=epsilon,
        min_samples=min_objects,
        metric='precomputed',
        n_jobs=-1,
    )
    labels = model.fit_predict(D_db)

    clusters: List[Set[int]] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = {ids[i] for i, lab in enumerate(labels) if lab == label}
        if len(members) >= min_objects:
            clusters.append(members)

    return clusters


# ============================================================
# Filter stage
# ============================================================

def build_partition_segments(
    simplified_segments: Dict[int, List[dict]],
    p_start: pd.Timestamp,
    p_end: pd.Timestamp,
) -> Dict[int, List[dict]]:
    """
    Put every simplified segment whose time interval intersects the partition
    into that partition. A boundary segment is intentionally included in both
    neighboring partitions, matching the paper's boundary treatment.
    """
    out: Dict[int, List[dict]] = {}
    for taxi_id, segs in simplified_segments.items():
        qualifying = [
            s for s in segs
            if s['t1'] >= p_start and s['t0'] <= p_end
        ]
        if qualifying:
            out[taxi_id] = qualifying
    return out


def merge_filter_candidates(
    previous: List[dict],
    clusters: List[Set[int]],
    p_start: pd.Timestamp,
    p_end: pd.Timestamp,
    partition_points: int,
    min_objects: int,
    lifetime_unit_points: int,
) -> Tuple[List[dict], List[dict]]:
    """Advance CuTS candidate state across one partition."""
    next_candidates: List[dict] = []
    assigned_cluster_indices: Set[int] = set()

    for cand in previous:
        matched = False
        for ci, cluster in enumerate(clusters):
            common = cand['objects'] & cluster
            if len(common) >= min_objects:
                matched = True
                assigned_cluster_indices.add(ci)
                next_candidates.append(
                    dict(
                        objects=set(common),
                        start_time=cand['start_time'],
                        end_time=p_end,
                        lifetime_points=cand['lifetime_points'] + lifetime_unit_points,
                    )
                )

        # If a candidate cannot be continued, emit it when its lifetime meets k.
        if not matched:
            # handled below via candidates returned by caller; store as finalized
            pass

    # New clusters become new candidates.
    for ci, cluster in enumerate(clusters):
        if ci not in assigned_cluster_indices:
            next_candidates.append(
                dict(
                    objects=set(cluster),
                    start_time=p_start,
                    end_time=p_end,
                    lifetime_points=partition_points,
                )
            )

    # Determine finalized candidates from previous state.
    finalized: List[dict] = []
    for cand in previous:
        still_matches = any(len(cand['objects'] & c) >= min_objects for c in clusters)
        if not still_matches and cand['lifetime_points'] >= partition_points:
            finalized.append(cand)

    return next_candidates, finalized


def cuts_filter(
    df_window: pd.DataFrame,
    epsilon: float,
    min_objects: int,
    min_lifetime_points: int,
    delta: float,
    lambda_points: int,
) -> Tuple[List[dict], Dict[int, List[dict]], Dict[int, pd.DataFrame], List[dict], Dict[str, float]]:
    """
    Run the CuTS filter stage on one 3-hour window.

    Timing is collected for each major operation in the filter so bottlenecks
    can be identified both cumulatively and on a per-partition basis.
    """
    function_start = time.perf_counter()

    # ------------------------------------------------------------
    # Timing accumulators
    # ------------------------------------------------------------
    timing = {
        'timestamp_setup': 0.0,
        'douglas_peucker': 0.0,
        'simplification_stats': 0.0,
        'boundary_setup': 0.0,
        'partition_build': 0.0,
        'traj_dbscan': 0.0,
        'candidate_matching': 0.0,
        'candidate_creation': 0.0,
        'state_deduplication': 0.0,
        'partition_bookkeeping': 0.0,
        'finalize_active': 0.0,
        'candidate_deduplication': 0.0,
    }

    # ------------------------------------------------------------
    # Timestamp setup
    # ------------------------------------------------------------
    step_start = time.perf_counter()
    timestamps = sorted(df_window['time'].dropna().unique())
    timing['timestamp_setup'] += time.perf_counter() - step_start

    if len(timestamps) < 2:
        total_time = time.perf_counter() - function_start
        timing['cuts_filter'] = 0.0
        timing['cuts_filter_total'] = total_time
        return [], {}, {}, [], timing

    simplified: Dict[int, pd.DataFrame] = {}
    simplified_segments: Dict[int, List[dict]] = {}

    # ------------------------------------------------------------
    # Douglas-Peucker simplification
    # ------------------------------------------------------------
    print(f"[CuTS] Simplifying {df_window['id'].nunique():,} taxi trajectories...")
    step_start = time.perf_counter()

    for taxi_id, group in df_window.groupby('id', sort=False):
        simp, segs = douglas_peucker_with_tolerance(group, delta)
        simplified[int(taxi_id)] = simp
        simplified_segments[int(taxi_id)] = segs

    simplification_time = time.perf_counter() - step_start
    timing['douglas_peucker'] += simplification_time

    # ------------------------------------------------------------
    # Simplification statistics
    # ------------------------------------------------------------
    step_start = time.perf_counter()
    orig_points = sum(len(g) for _, g in df_window.groupby('id'))
    simp_points = sum(len(g) for g in simplified.values())
    reduction = 100.0 * (1.0 - simp_points / max(orig_points, 1))
    timing['simplification_stats'] += time.perf_counter() - step_start

    print(f"[CuTS] Original points:   {orig_points:,}")
    print(f"[CuTS] Simplified points: {simp_points:,}")
    print(f"[CuTS] Reduction:         {reduction:.2f}%")
    print(f"[CuTS] Douglas-Peucker time: {simplification_time:.4f} seconds")

    # ------------------------------------------------------------
    # Partition boundary setup
    # ------------------------------------------------------------
    step_start = time.perf_counter()

    timestamps = [pd.Timestamp(t) for t in timestamps]
    window_start = timestamps[0]
    window_end = timestamps[-1]

    # CuTS partitions are defined by lambda consecutive time points.
    # Neighboring partitions share the boundary time point so a segment
    # crossing the boundary is not dropped.
    if lambda_points < 2:
        raise ValueError('lambda_points must be at least 2')

    boundaries: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    stride = lambda_points - 1
    start_idx = 0
    while start_idx < len(timestamps) - 1:
        end_idx = min(start_idx + lambda_points - 1, len(timestamps) - 1)
        boundaries.append((timestamps[start_idx], timestamps[end_idx]))
        if end_idx == len(timestamps) - 1:
            break
        start_idx += stride

    timing['boundary_setup'] += time.perf_counter() - step_start

    active: List[dict] = []
    completed: List[dict] = []
    filter_rows: List[dict] = []

    # The original CuTS filter timer starts after simplification/setup.
    filter_start = time.perf_counter()

    for p_idx, (p_start, p_end) in enumerate(boundaries, start=1):
        partition_start = time.perf_counter()

        # --------------------------------------------------------
        # Build the simplified segments used by this partition
        # --------------------------------------------------------
        step_start = time.perf_counter()
        part = build_partition_segments(simplified_segments, p_start, p_end)
        partition_build_time = time.perf_counter() - step_start
        timing['partition_build'] += partition_build_time

        # --------------------------------------------------------
        # TRAJ-DBSCAN clustering
        # --------------------------------------------------------
        step_start = time.perf_counter()
        clusters = traj_dbscan(
            part,
            epsilon=epsilon,
            min_objects=min_objects,
            window_start=p_start,
            window_end=p_end,
        )
        traj_dbscan_time = time.perf_counter() - step_start
        timing['traj_dbscan'] += traj_dbscan_time

        next_active: List[dict] = []
        matched_prev: Set[int] = set()
        assigned_clusters: Set[int] = set()

        # --------------------------------------------------------
        # Match existing candidates to clusters
        # --------------------------------------------------------
        step_start = time.perf_counter()

        for prev_idx, cand in enumerate(active):
            any_match = False
            for ci, cluster in enumerate(clusters):
                common = cand['objects'] & cluster
                if len(common) >= min_objects:
                    any_match = True
                    matched_prev.add(prev_idx)
                    assigned_clusters.add(ci)
                    next_active.append(
                        dict(
                            objects=set(common),
                            start_time=cand['start_time'],
                            end_time=p_end,
                            lifetime_points=cand['lifetime_points'] + lambda_points,
                        )
                    )
            if not any_match and cand['lifetime_points'] >= min_lifetime_points:
                completed.append(cand)

        candidate_matching_time = time.perf_counter() - step_start
        timing['candidate_matching'] += candidate_matching_time

        # --------------------------------------------------------
        # Start candidates from previously unused clusters
        # --------------------------------------------------------
        step_start = time.perf_counter()

        for ci, cluster in enumerate(clusters):
            if ci not in assigned_clusters and len(cluster) >= min_objects:
                next_active.append(
                    dict(
                        objects=set(cluster),
                        start_time=p_start,
                        end_time=p_end,
                        lifetime_points=lambda_points,
                    )
                )

        candidate_creation_time = time.perf_counter() - step_start
        timing['candidate_creation'] += candidate_creation_time

        # --------------------------------------------------------
        # De-duplicate candidate states
        # --------------------------------------------------------
        step_start = time.perf_counter()

        unique = {}
        for cand in next_active:
            key = (
                tuple(sorted(cand['objects'])),
                cand['start_time'],
                cand['end_time'],
                cand['lifetime_points'],
            )
            unique[key] = cand
        active = list(unique.values())

        state_dedup_time = time.perf_counter() - step_start
        timing['state_deduplication'] += state_dedup_time

        # --------------------------------------------------------
        # Partition bookkeeping + reporting
        # --------------------------------------------------------
        step_start = time.perf_counter()

        partition_elapsed_before_bookkeeping = time.perf_counter() - partition_start

        filter_rows.append({
            'partition': p_idx,
            'start_time': p_start,
            'end_time': p_end,
            'objects_in_partition': len(part),
            'filter_clusters': len(clusters),
            'active_candidates': len(active),
            'time_partition_build': partition_build_time,
            'time_traj_dbscan': traj_dbscan_time,
            'time_candidate_matching': candidate_matching_time,
            'time_candidate_creation': candidate_creation_time,
            'time_state_deduplication': state_dedup_time,
            'time_core_partition': partition_elapsed_before_bookkeeping,
        })

        print(
            f"[CuTS] Partition {p_idx:02d}/{len(boundaries):02d} "
            f"{p_start:%H:%M}–{p_end:%H:%M}: "
            f"{len(part):,} taxis, {len(clusters)} clusters, {len(active)} active candidates"
        )
        print(
            f"       timing: build={partition_build_time:.4f}s | "
            f"dbscan={traj_dbscan_time:.4f}s | "
            f"match={candidate_matching_time:.4f}s | "
            f"new={candidate_creation_time:.4f}s | "
            f"dedup={state_dedup_time:.4f}s"
        )

        bookkeeping_time = time.perf_counter() - step_start
        timing['partition_bookkeeping'] += bookkeeping_time
        filter_rows[-1]['time_bookkeeping'] = bookkeeping_time
        filter_rows[-1]['time_partition_total'] = time.perf_counter() - partition_start

    # ------------------------------------------------------------
    # Finalize candidates that survive the final partition
    # ------------------------------------------------------------
    step_start = time.perf_counter()
    completed.extend([c for c in active if c['lifetime_points'] >= min_lifetime_points])
    timing['finalize_active'] += time.perf_counter() - step_start

    # ------------------------------------------------------------
    # Final candidate de-duplication
    # ------------------------------------------------------------
    step_start = time.perf_counter()

    deduped = {}
    for c in completed:
        key = (
            tuple(sorted(c['objects'])),
            pd.Timestamp(c['start_time']).round('10min'),
            pd.Timestamp(c['end_time']).round('10min'),
        )
        # Keep the longest instance if duplicates occur.
        old = deduped.get(key)
        if old is None or c['lifetime_points'] > old['lifetime_points']:
            deduped[key] = c

    candidates = list(deduped.values())
    timing['candidate_deduplication'] += time.perf_counter() - step_start

    filtering_time = time.perf_counter() - filter_start
    total_function_time = time.perf_counter() - function_start

    timing['cuts_filter'] = filtering_time
    timing['cuts_filter_total'] = total_function_time

    print(f"[CuTS] Filter candidates: {len(candidates)}")
    print(f"[CuTS] Filter time: {filtering_time:.4f} seconds")

    print("\n[CuTS] Detailed cuts_filter timing:")
    timing_order = [
        ('timestamp_setup', 'Timestamp setup'),
        ('douglas_peucker', 'Douglas-Peucker'),
        ('simplification_stats', 'Simplification stats'),
        ('boundary_setup', 'Boundary setup'),
        ('partition_build', 'Build partition segments'),
        ('traj_dbscan', 'TRAJ-DBSCAN'),
        ('candidate_matching', 'Candidate matching'),
        ('candidate_creation', 'Candidate creation'),
        ('state_deduplication', 'State de-duplication'),
        ('partition_bookkeeping', 'Partition bookkeeping'),
        ('finalize_active', 'Finalize active'),
        ('candidate_deduplication', 'Candidate de-duplication'),
    ]

    for key, label in timing_order:
        seconds = timing[key]
        percent = 100.0 * seconds / total_function_time if total_function_time > 0 else 0.0
        print(f"       {label:<28} {seconds:>10.4f}s  {percent:>6.2f}%")

    print(f"       {'Total cuts_filter()':<28} {total_function_time:>10.4f}s  {100.0:>6.2f}%")

    cuts_times = timing.copy()

    return candidates, simplified_segments, simplified, filter_rows, cuts_times


# ============================================================
# Refinement: CMC-style exact snapshot test
# ============================================================

def interpolate_candidate_points(
    df: pd.DataFrame,
    object_ids: Iterable[int],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    freq: str = '10min',
) -> pd.DataFrame:
    """Interpolate missing observations for candidate taxis on the regular grid."""
    times = pd.date_range(start_time, end_time, freq=freq)
    out = []

    for taxi_id in sorted(set(int(x) for x in object_ids)):
        g = df[(df['id'] == taxi_id) & (df['time'] >= start_time) & (df['time'] <= end_time)]
        if g.empty:
            continue

        g = g[['time', 'x', 'y']].drop_duplicates('time').set_index('time').sort_index()
        g = g.reindex(g.index.union(times)).sort_index()
        g[['x', 'y']] = g[['x', 'y']].interpolate(method='time', limit_area='inside')
        g = g.reindex(times)
        g['id'] = taxi_id
        g['time'] = g.index
        g = g.dropna(subset=['x', 'y']).reset_index(drop=True)
        out.append(g[['id', 'time', 'x', 'y']])

    if not out:
        return pd.DataFrame(columns=['id', 'time', 'x', 'y'])
    return pd.concat(out, ignore_index=True)


def snapshot_clusters(
    snapshot: pd.DataFrame,
    epsilon: float,
    min_objects: int,
) -> List[Set[int]]:
    if len(snapshot) < min_objects:
        return []

    xy = snapshot[['x', 'y']].to_numpy(dtype=float)
    model = DBSCAN(eps=epsilon, min_samples=min_objects, metric='euclidean', n_jobs=-1)
    labels = model.fit_predict(xy)

    clusters = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = set(snapshot.loc[labels == label, 'id'].astype(int).tolist())
        if len(members) >= min_objects:
            clusters.append(members)
    return clusters


def refine_candidate(
    df: pd.DataFrame,
    candidate: dict,
    epsilon: float,
    min_objects: int,
    min_lifetime_points: int,
) -> List[dict]:
    """
    Refine one filter candidate using snapshot DBSCAN + intersection tracking.
    """
    points = interpolate_candidate_points(
        df,
        candidate['objects'],
        candidate['start_time'],
        candidate['end_time'],
        freq='10min',
    )
    if points.empty:
        return []

    times = sorted(points['time'].unique())
    active: List[dict] = []
    confirmed: List[dict] = []

    for t in times:
        snap = points[points['time'] == t]
        clusters = snapshot_clusters(snap, epsilon, min_objects)

        next_active = []
        used_cluster = set()

        for state in active:
            matched = False
            for ci, cluster in enumerate(clusters):
                common = state['objects'] & cluster
                if len(common) >= min_objects:
                    matched = True
                    used_cluster.add(ci)
                    next_active.append({
                        'objects': common,
                        'start_time': state['start_time'],
                        'end_time': pd.Timestamp(t),
                        'lifetime_points': state['lifetime_points'] + 1,
                    })

            if not matched and state['lifetime_points'] >= min_lifetime_points:
                confirmed.append(state)

        for ci, cluster in enumerate(clusters):
            if ci not in used_cluster:
                next_active.append({
                    'objects': set(cluster),
                    'start_time': pd.Timestamp(t),
                    'end_time': pd.Timestamp(t),
                    'lifetime_points': 1,
                })

        # Remove states that are strict duplicates.
        unique = {}
        for state in next_active:
            key = (
                tuple(sorted(state['objects'])),
                state['start_time'],
                state['end_time'],
                state['lifetime_points'],
            )
            unique[key] = state
        active = list(unique.values())

    confirmed.extend([s for s in active if s['lifetime_points'] >= min_lifetime_points])

    # Convert to final convoy records and only keep convoys that overlap the
    # candidate's object set, since this is a refinement of that candidate.
    results = []
    seen = set()
    for state in confirmed:
        objects = tuple(sorted(int(x) for x in state['objects']))
        if len(objects) < min_objects:
            continue
        key = (objects, state['start_time'], state['end_time'])
        if key in seen:
            continue
        seen.add(key)
        results.append({
            'objects': list(objects),
            'start_time': pd.Timestamp(state['start_time']),
            'end_time': pd.Timestamp(state['end_time']),
            'lifetime_points': int(state['lifetime_points']),
            'lifetime_minutes': int(state['lifetime_points'] * 10),
        })

    return results


def refine_all_candidates(
    df: pd.DataFrame,
    candidates: List[dict],
    epsilon: float,
    min_objects: int,
    min_lifetime_points: int,
) -> List[dict]:
    refined = []
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"[Refine] Candidate {idx}/{len(candidates)}: "
            f"{len(candidate['objects'])} taxis, "
            f"{candidate['start_time']:%H:%M}–{candidate['end_time']:%H:%M}"
        )
        refined.extend(
            refine_candidate(
                df,
                candidate,
                epsilon,
                min_objects,
                min_lifetime_points,
            )
        )

    # Global de-duplication.
    unique = {}
    for c in refined:
        key = (tuple(c['objects']), c['start_time'], c['end_time'])
        unique[key] = c

    results = list(unique.values())
    results.sort(key=lambda x: (x['start_time'], -len(x['objects'])))
    return results


# ==========================================
# Example Usage with T-Drive Data
# ==========================================
# ==========================================
# Example Usage with T-Drive Data
# ==========================================

if __name__ == "__main__":

    log_filename = time.strftime("cuts_terminal_output_%Y%m%d_%H%M%S.txt")
    log_file = open(log_filename, 'w', encoding='utf-8', buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:

        file = 'tdrive_processed.csv'

        program_start = time.time()
        stage_times = {}

        # Stage 1: Load data
        stage_start = time.time()

        print("Loading data...")
        df_temp = pd.read_csv(
            file,
            header=0,
            parse_dates=['time']
        )

        stage_times['load_data'] = time.time() - stage_start


        # Stage 2: Select time window and clean data
        stage_start = time.time()

        start_time = pd.to_datetime("2008-02-02 13:00:00")
        end_time = pd.to_datetime("2008-02-02 18:00:00")

        df_temp = df_temp[
            (df_temp['time'] >= start_time)
            & (df_temp['time'] <= end_time)
        ].copy()

        df_temp = (
            df_temp.sort_values(['id', 'time'])
            .drop_duplicates(['id', 'time'])
            .reset_index(drop=True)
        )

        stage_times['window_setup'] = time.time() - stage_start

        print(f"Total rows in time window: {len(df_temp)}")
        print(f"Total unique taxis in window: {df_temp['id'].nunique()}")


        # Stage 3: Convert longitude/latitude to meters
        stage_start = time.time()

        ref_lat = float(df_temp['y'].mean())
        lon_scale = 111320.0 * math.cos(math.radians(ref_lat))
        lat_scale = 111320.0

        lon0 = float(df_temp['x'].mean())
        lat0 = float(df_temp['y'].mean())

        df_metric = df_temp.copy()
        df_metric['x'] = (df_metric['x'] - lon0) * lon_scale
        df_metric['y'] = (df_metric['y'] - lat0) * lat_scale

        stage_times['coordinate_conversion'] = time.time() - stage_start


        # Stage 4 + 5: Douglas-Peucker and CuTS filter
        print("Running CuTS Algorithm...")

        candidates, simplified_segments, simplified, filter_rows, cuts_times = cuts_filter(
            df_metric,
            epsilon=total_dist,
            min_objects=total_obj,
            min_lifetime_points=total_time,
            delta=dp_tolerance,
            lambda_points=lambda_points
        )

        stage_times.update(cuts_times)


        # Stage 6: Refinement
        stage_start = time.time()

        convoys = refine_all_candidates(
            df_metric,
            candidates,
            epsilon=total_dist,
            min_objects=total_obj,
            min_lifetime_points=total_time
        )

        stage_times['refinement'] = time.time() - stage_start


        # Stage 7: Save CSV
        stage_start = time.time()

        df_convoys = pd.DataFrame(convoys)
        df_convoys.to_csv(
            f"discovered_convoys_cuts_{total_obj}_{total_time}_{total_dist}.csv",
            index=False
        )

        stage_times['save_csv'] = time.time() - stage_start


        print(f"\nTotal Convoys Discovered: {len(convoys)}")

        for i, c in enumerate(convoys):
            print(
                f"Convoy {i+1}: "
                f"Objects {c['objects']} | "
                f"Time: {c['start_time']} to {c['end_time']}"
            )


        # Stage 8: Visualization
        stage_start = time.time()

        if convoys:
            viz.visualize_convoys(
                df_temp,
                convoys,
                start_time,
                end_time,
                file=f"all_convoys_map_beijing_cuts_{total_obj}_{total_time}_{total_dist}"
            )

        stage_times['visualization'] = time.time() - stage_start


        # Timing summary
        total_runtime = time.time() - program_start

        print("\n======================================================")
        print("                 TIMING RESULTS")
        print("======================================================")
        print(f"{'Stage':<30}{'Seconds':>12}{'Percent':>12}")
        print("-" * 54)

        stage_names = [
            ('load_data', 'Load data'),
            ('window_setup', 'Window setup'),
            ('coordinate_conversion', 'Coordinate conversion'),
            ('douglas_peucker', 'Douglas-Peucker'),
            ('cuts_filter', 'CuTS filter'),
            ('refinement', 'Refinement'),
            ('save_csv', 'Save CSV'),
            ('visualization', 'Visualization')
        ]

        for key, label in stage_names:
            seconds = stage_times.get(key, 0.0)
            percent = 100.0 * seconds / total_runtime if total_runtime > 0 else 0.0

            print(
                f"{label:<30}"
                f"{seconds:>12.4f}"
                f"{percent:>11.2f}%"
            )

        print("-" * 54)
        print(f"{'TOTAL PROGRAM TIME':<30}{total_runtime:>12.4f}")
        print("======================================================")

        print(f"\nTerminal output saved to: {log_filename}")
    finally:
        # Flush the duplicated streams before restoring the real terminal streams.
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
