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
    7. Save CSV results and interactive Folium maps.

Expected preprocessed CSV columns:
    id,time,x,y
where x=longitude and y=latitude.

Install if needed:
    pip install pandas numpy scipy scikit-learn folium matplotlib
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


# ============================================================
# Data structures
# ============================================================

@dataclass
class Segment:
    taxi_id: int
    t0: pd.Timestamp
    t1: pd.Timestamp
    x0: float
    y0: float
    x1: float
    y1: float
    tolerance: float


@dataclass
class FilterCandidate:
    objects: Set[int]
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    lifetime_points: int


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


def segments_intersect(a: Segment, b: Segment) -> bool:
    o1 = orientation(a.x0, a.y0, a.x1, a.y1, b.x0, b.y0)
    o2 = orientation(a.x0, a.y0, a.x1, a.y1, b.x1, b.y1)
    o3 = orientation(b.x0, b.y0, b.x1, b.y1, a.x0, a.y0)
    o4 = orientation(b.x0, b.y0, b.x1, b.y1, a.x1, a.y1)

    if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and \
       ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)):
        return True

    if abs(o1) < 1e-12 and on_segment(a.x0, a.y0, a.x1, a.y1, b.x0, b.y0):
        return True
    if abs(o2) < 1e-12 and on_segment(a.x0, a.y0, a.x1, a.y1, b.x1, b.y1):
        return True
    if abs(o3) < 1e-12 and on_segment(b.x0, b.y0, b.x1, b.y1, a.x0, a.y0):
        return True
    if abs(o4) < 1e-12 and on_segment(b.x0, b.y0, b.x1, b.y1, a.x1, a.y1):
        return True
    return False


def segment_distance(a: Segment, b: Segment) -> float:
    """Minimum 2D Euclidean distance between two line segments."""
    if segments_intersect(a, b):
        return 0.0

    return min(
        point_to_segment_distance(a.x0, a.y0, b.x0, b.y0, b.x1, b.y1),
        point_to_segment_distance(a.x1, a.y1, b.x0, b.y0, b.x1, b.y1),
        point_to_segment_distance(b.x0, b.y0, a.x0, a.y0, a.x1, a.y1),
        point_to_segment_distance(b.x1, b.y1, a.x0, a.y0, a.x1, a.y1),
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


def douglas_peucker_with_tolerance(group: pd.DataFrame, delta: float) -> Tuple[pd.DataFrame, List[Segment]]:
    """Simplify one trajectory and calculate each segment's actual max DP error."""
    g = group.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    if len(g) < 2:
        return g.copy(), []

    points = g[['x', 'y']].to_numpy(dtype=float)
    idx = np.arange(len(g), dtype=int)
    keep = _dp_recursive(points, idx, delta)
    keep = sorted(set(keep))

    simp = g.iloc[keep].copy().reset_index(drop=True)
    segments: List[Segment] = []

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
            Segment(
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

def segment_interval_overlap(a: Segment, b: Segment,
                              window_start: pd.Timestamp,
                              window_end: pd.Timestamp) -> bool:
    a0 = max(a.t0, window_start)
    a1 = min(a.t1, window_end)
    b0 = max(b.t0, window_start)
    b1 = min(b.t1, window_end)
    return max(a0, b0) <= min(a1, b1)


def trajectory_filter_distance(segments_a: Sequence[Segment],
                               segments_b: Sequence[Segment],
                               window_start: pd.Timestamp,
                               window_end: pd.Timestamp) -> float:
    """
    CuTS-style omega bound:
        min(D_LL(segment_a, segment_b) - tol_a - tol_b)
    over temporally overlapping segment pairs.
    """
    best = float('inf')
    for a in segments_a:
        if a.t1 < window_start or a.t0 > window_end:
            continue
        for b in segments_b:
            if b.t1 < window_start or b.t0 > window_end:
                continue
            if not segment_interval_overlap(a, b, window_start, window_end):
                continue
            d = max(0.0, segment_distance(a, b) - a.tolerance - b.tolerance)
            best = min(best, d)
            if best <= 0.0:
                return 0.0
    return best


def build_trajectory_distance_matrix(
    partition_segments: Dict[int, List[Segment]],
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


def traj_dbscan(partition_segments: Dict[int, List[Segment]],
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
    simplified_segments: Dict[int, List[Segment]],
    p_start: pd.Timestamp,
    p_end: pd.Timestamp,
) -> Dict[int, List[Segment]]:
    """
    Put every simplified segment whose time interval intersects the partition
    into that partition. A boundary segment is intentionally included in both
    neighboring partitions, matching the paper's boundary treatment.
    """
    out: Dict[int, List[Segment]] = {}
    for taxi_id, segs in simplified_segments.items():
        qualifying = [
            s for s in segs
            if s.t1 >= p_start and s.t0 <= p_end
        ]
        if qualifying:
            out[taxi_id] = qualifying
    return out


def merge_filter_candidates(
    previous: List[FilterCandidate],
    clusters: List[Set[int]],
    p_start: pd.Timestamp,
    p_end: pd.Timestamp,
    partition_points: int,
    min_objects: int,
    lifetime_unit_points: int,
) -> Tuple[List[FilterCandidate], List[FilterCandidate]]:
    """Advance CuTS candidate state across one partition."""
    next_candidates: List[FilterCandidate] = []
    assigned_cluster_indices: Set[int] = set()

    for cand in previous:
        matched = False
        for ci, cluster in enumerate(clusters):
            common = cand.objects & cluster
            if len(common) >= min_objects:
                matched = True
                assigned_cluster_indices.add(ci)
                next_candidates.append(
                    FilterCandidate(
                        objects=set(common),
                        start_time=cand.start_time,
                        end_time=p_end,
                        lifetime_points=cand.lifetime_points + lifetime_unit_points,
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
                FilterCandidate(
                    objects=set(cluster),
                    start_time=p_start,
                    end_time=p_end,
                    lifetime_points=partition_points,
                )
            )

    # Determine finalized candidates from previous state.
    finalized: List[FilterCandidate] = []
    for cand in previous:
        still_matches = any(len(cand.objects & c) >= min_objects for c in clusters)
        if not still_matches and cand.lifetime_points >= partition_points:
            finalized.append(cand)

    return next_candidates, finalized


def cuts_filter(
    df_window: pd.DataFrame,
    epsilon: float,
    min_objects: int,
    min_lifetime_points: int,
    delta: float,
    lambda_points: int,
) -> Tuple[List[FilterCandidate], Dict[int, List[Segment]], Dict[int, pd.DataFrame], List[dict]]:
    """
    Run the CuTS filter stage on one 3-hour window.
    """
    timestamps = sorted(df_window['time'].dropna().unique())
    if len(timestamps) < 2:
        return [], {}, {}, []

    simplified: Dict[int, pd.DataFrame] = {}
    simplified_segments: Dict[int, List[Segment]] = {}

    print(f"[CuTS] Simplifying {df_window['id'].nunique():,} taxi trajectories...")
    for taxi_id, group in df_window.groupby('id', sort=False):
        simp, segs = douglas_peucker_with_tolerance(group, delta)
        simplified[int(taxi_id)] = simp
        simplified_segments[int(taxi_id)] = segs

    orig_points = sum(len(g) for _, g in df_window.groupby('id'))
    simp_points = sum(len(g) for g in simplified.values())
    reduction = 100.0 * (1.0 - simp_points / max(orig_points, 1))
    print(f"[CuTS] Original points:   {orig_points:,}")
    print(f"[CuTS] Simplified points: {simp_points:,}")
    print(f"[CuTS] Reduction:         {reduction:.2f}%")

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

    active: List[FilterCandidate] = []
    completed: List[FilterCandidate] = []
    filter_rows: List[dict] = []

    for p_idx, (p_start, p_end) in enumerate(boundaries, start=1):
        part = build_partition_segments(simplified_segments, p_start, p_end)
        clusters = traj_dbscan(
            part,
            epsilon=epsilon,
            min_objects=min_objects,
            window_start=p_start,
            window_end=p_end,
        )

        next_active: List[FilterCandidate] = []
        matched_prev: Set[int] = set()
        assigned_clusters: Set[int] = set()

        # Track each existing candidate into every compatible cluster.
        for prev_idx, cand in enumerate(active):
            any_match = False
            for ci, cluster in enumerate(clusters):
                common = cand.objects & cluster
                if len(common) >= min_objects:
                    any_match = True
                    matched_prev.add(prev_idx)
                    assigned_clusters.add(ci)
                    next_active.append(
                        FilterCandidate(
                            objects=set(common),
                            start_time=cand.start_time,
                            end_time=p_end,
                            lifetime_points=cand.lifetime_points + lambda_points,
                        )
                    )
            if not any_match and cand.lifetime_points >= min_lifetime_points:
                completed.append(cand)

        # Start candidates from clusters not used by a previous candidate.
        for ci, cluster in enumerate(clusters):
            if ci not in assigned_clusters and len(cluster) >= min_objects:
                next_active.append(
                    FilterCandidate(
                        objects=set(cluster),
                        start_time=p_start,
                        end_time=p_end,
                        lifetime_points=lambda_points,
                    )
                )

        # De-duplicate states created by multiple equivalent transitions.
        unique = {}
        for cand in next_active:
            key = (tuple(sorted(cand.objects)), cand.start_time, cand.end_time, cand.lifetime_points)
            unique[key] = cand
        active = list(unique.values())

        filter_rows.append({
            'partition': p_idx,
            'start_time': p_start,
            'end_time': p_end,
            'objects_in_partition': len(part),
            'filter_clusters': len(clusters),
            'active_candidates': len(active),
        })

        print(
            f"[CuTS] Partition {p_idx:02d}/{len(boundaries):02d} "
            f"{p_start:%H:%M}–{p_end:%H:%M}: "
            f"{len(part):,} taxis, {len(clusters)} clusters, {len(active)} active candidates"
        )

    # Finalize candidates that survive through the final partition.
    completed.extend([c for c in active if c.lifetime_points >= min_lifetime_points])

    # De-duplicate candidates.
    deduped = {}
    for c in completed:
        key = (
            tuple(sorted(c.objects)),
            pd.Timestamp(c.start_time).round('10min'),
            pd.Timestamp(c.end_time).round('10min'),
        )
        # Keep the longest instance if duplicates occur.
        old = deduped.get(key)
        if old is None or c.lifetime_points > old.lifetime_points:
            deduped[key] = c

    candidates = list(deduped.values())
    print(f"[CuTS] Filter candidates: {len(candidates)}")

    return candidates, simplified_segments, simplified, filter_rows


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
    candidate: FilterCandidate,
    epsilon: float,
    min_objects: int,
    min_lifetime_points: int,
) -> List[dict]:
    """
    Refine one filter candidate using snapshot DBSCAN + intersection tracking.
    """
    points = interpolate_candidate_points(
        df,
        candidate.objects,
        candidate.start_time,
        candidate.end_time,
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
    candidates: List[FilterCandidate],
    epsilon: float,
    min_objects: int,
    min_lifetime_points: int,
) -> List[dict]:
    refined = []
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"[Refine] Candidate {idx}/{len(candidates)}: "
            f"{len(candidate.objects)} taxis, "
            f"{candidate.start_time:%H:%M}–{candidate.end_time:%H:%M}"
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


# ============================================================
# Window selection / plotting
# ============================================================

def choose_window(df: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    window = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy()
    return window, start_time, end_time


def normalize_day_string(df: pd.DataFrame, day: Optional[str]) -> str:
    if day:
        return day
    return pd.Timestamp(df['time'].min()).strftime('%Y-%m-%d')


def plot_overview(df_window: pd.DataFrame,
                  convoys: List[dict],
                  out_file: str) -> None:
    """Matplotlib overview of convoy trajectories in the selected window."""
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(
        df_window['x'], df_window['y'],
        s=4, alpha=0.10, label='All taxi points'
    )

    for idx, convoy in enumerate(convoys):
        cdf = df_window[
            df_window['id'].isin(convoy['objects']) &
            (df_window['time'] >= convoy['start_time']) &
            (df_window['time'] <= convoy['end_time'])
        ].sort_values(['id', 'time'])

        for taxi_id, g in cdf.groupby('id'):
            ax.plot(g['x'], g['y'], linewidth=2.5, alpha=0.8)

        cx = cdf['x'].mean() if not cdf.empty else np.nan
        cy = cdf['y'].mean() if not cdf.empty else np.nan
        if np.isfinite(cx) and np.isfinite(cy):
            ax.text(
                cx, cy,
                f"C{idx + 1} ({len(convoy['objects'])})",
                fontsize=9,
                fontweight='bold'
            )

    ax.set_title('CuTS Convoys — T-drive 3-hour window')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved: {out_file}")


def save_folium_map(df: pd.DataFrame, convoys: List[dict], out_file: str) -> None:
    """Interactive Beijing map, compatible with the user's existing visualization style."""
    if not convoys:
        print('[Map] No refined convoys to visualize.')
        return

    active_objects = sorted(set(obj for c in convoys for obj in c['objects']))
    active_df = df[df['id'].isin(active_objects)]

    center_lat = float(active_df['y'].mean()) if not active_df.empty else 39.9
    center_lon = float(active_df['x'].mean()) if not active_df.empty else 116.4

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    colors = [
        'red', 'blue', 'green', 'purple', 'orange',
        'darkred', 'cadetblue', 'darkgreen', 'pink', 'black'
    ]

    for idx, convoy in enumerate(convoys):
        color = colors[idx % len(colors)]
        label = (
            f"Convoy {idx + 1} | size={len(convoy['objects'])} | "
            f"{convoy['start_time']:%H:%M}–{convoy['end_time']:%H:%M}"
        )
        group = folium.FeatureGroup(name=label)

        cdf = df[
            df['id'].isin(convoy['objects']) &
            (df['time'] >= convoy['start_time']) &
            (df['time'] <= convoy['end_time'])
        ].sort_values(['id', 'time'])

        for taxi_id, g in cdf.groupby('id'):
            route = list(zip(g['y'], g['x']))
            if not route:
                continue

            folium.PolyLine(
                route,
                weight=4,
                color=color,
                opacity=0.8,
                tooltip=f"Convoy {idx + 1} | Taxi {taxi_id}"
            ).add_to(group)

            folium.CircleMarker(
                location=route[0],
                radius=4,
                color=color,
                fill=True,
                fill_color='white',
                fill_opacity=1.0,
                tooltip=f"Start | Taxi {taxi_id}"
            ).add_to(group)

            folium.Marker(
                location=route[-1],
                icon=folium.Icon(color=color, icon='flag'),
                tooltip=f"End | Taxi {taxi_id}"
            ).add_to(group)

        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_file)
    print(f"[Map] Saved: {out_file}")


# ============================================================
# Main
# ============================================================

def run_cuts(
    csv_path: str,
    day: Optional[str],
    start_time: str,
    epsilon: float,
    min_objects: int,
    min_lifetime_points: int,
    delta: float,
    lambda_points: int,
    output_prefix: str,
) -> List[dict]:
    print(f'[Load] Reading {csv_path}')
    df = pd.read_csv(csv_path)

    required = {'id', 'time', 'x', 'y'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    df['time'] = pd.to_datetime(df['time'])
    df['id'] = df['id'].astype(int)
    df = df.sort_values(['id', 'time']).drop_duplicates(['id', 'time']).reset_index(drop=True)

    # Match the CMC example exactly: 2008-02-02 13:00 through 18:00.
    # This is a 5-hour window, even though the earlier CuTS example used 3 hours.
    day = normalize_day_string(df, day)
    start = pd.Timestamp(f'{day} {start_time}')
    end = start + pd.Timedelta(hours=5)
    window_geo, start, end = choose_window(df, start, end)

    if window_geo.empty:
        raise ValueError(
            f'No points found in requested window {start} to {end}. '
            f'Choose another day/start time.'
        )

    # Convert lon/lat to a local metric coordinate system for all spatial
    # computations. Keep window_geo unchanged for visualization.
    ref_lat = float(window_geo['y'].mean())
    lon_scale = 111320.0 * math.cos(math.radians(ref_lat))
    lat_scale = 111320.0

    window_df = window_geo.copy()
    lon0 = float(window_df['x'].mean())
    lat0 = float(window_df['y'].mean())
    window_df['x'] = (window_df['x'] - lon0) * lon_scale
    window_df['y'] = (window_df['y'] - lat0) * lat_scale

    # Require at least two snapshots for a trajectory segment.
    print(f'\n[Window] {start} -> {end}')
    print(f'[Window] Points: {len(window_df):,}')
    print(f'[Window] Taxis:  {window_df["id"].nunique():,}')
    print(f'[Window] Snapshots present: {window_df["time"].nunique():,}')
    print(
        f'[Params] epsilon={epsilon} m, m={min_objects}, '
        f'k={min_lifetime_points} points, delta={delta} m, '
        f'lambda={lambda_points} points'
    )

    candidates, simplified_segments, simplified, filter_log = cuts_filter(
        window_df,
        epsilon=epsilon,
        min_objects=min_objects,
        min_lifetime_points=min_lifetime_points,
        delta=delta,
        lambda_points=lambda_points,
    )

    # Refinement operates on the same window used by the CMC example.
    convoys = refine_all_candidates(
        window_df,
        candidates,
        epsilon=epsilon,
        min_objects=min_objects,
        min_lifetime_points=min_lifetime_points,
    )

    # Save convoy summary.
    convoy_rows = []
    for idx, c in enumerate(convoys, start=1):
        convoy_rows.append({
            'convoy_id': idx,
            'object_count': len(c['objects']),
            'objects': ','.join(map(str, c['objects'])),
            'start_time': c['start_time'],
            'end_time': c['end_time'],
            'lifetime_points': c['lifetime_points'],
            'lifetime_minutes': c['lifetime_minutes'],
        })

    convoy_csv = f'{output_prefix}_convoys.csv'
    pd.DataFrame(convoy_rows).to_csv(convoy_csv, index=False)
    print(f'[Output] Convoy CSV: {convoy_csv}')

    filter_csv = f'{output_prefix}_filter_log.csv'
    pd.DataFrame(filter_log).to_csv(filter_csv, index=False)
    print(f'[Output] Filter log: {filter_csv}')

    overview_png = f'{output_prefix}_overview.png'
    plot_overview(window_geo, convoys, overview_png)

    map_html = f'{output_prefix}_map.html'
    save_folium_map(window_geo, convoys, map_html)

    print('\n==================== FINAL RESULTS ====================')
    print(f'Convoys discovered: {len(convoys)}')
    for idx, c in enumerate(convoys, start=1):
        print(
            f"  Convoy {idx}: {len(c['objects'])} taxis | "
            f"{c['start_time']:%Y-%m-%d %H:%M} -> {c['end_time']:%H:%M} | "
            f"{c['lifetime_minutes']} min | objects={c['objects']}"
        )
    print('=======================================================')

    return convoys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CuTS convoy discovery for T-drive')
    parser.add_argument('--csv', default='tdrive_processed.csv', help='Preprocessed T-drive CSV')
    parser.add_argument('--day', default='2008-02-02', help='Day, e.g. 2008-02-02.')
    parser.add_argument('--start', default='13:00', help='Window start time')
    parser.add_argument('--epsilon', type=float, default=100.0, help='e: CuTS/DBSCAN distance threshold in meters')
    parser.add_argument('--min-objects', type=int, default=3, help='m: minimum convoy size')
    parser.add_argument('--min-lifetime-points', type=int, default=3, help='k: minimum lifetime in 10-minute snapshots')
    parser.add_argument('--delta', type=float, default=100.0, help='Douglas-Peucker tolerance in meters')
    parser.add_argument('--lambda-points', type=int, default=3, help='CuTS partition size in 10-minute points')
    parser.add_argument('--output-prefix', default='discovered_convoys_3_3_100', help='Output filename prefix')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_cuts(
        csv_path=args.csv,
        day=args.day,
        start_time=args.start,
        epsilon=args.epsilon,
        min_objects=args.min_objects,
        min_lifetime_points=args.min_lifetime_points,
        delta=args.delta,
        lambda_points=args.lambda_points,
        output_prefix=args.output_prefix,
    )
