import time as time
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

import visualize_convoy_tdrive as viz
import cmc_example as cmc


# ============================================================
# 1. Coordinate Conversion
# ============================================================

def create_metric_coordinates(df):
    """
    Convert longitude/latitude coordinates into local metric
    coordinates centered around the dataset.

    Input:
        df['x'] = longitude
        df['y'] = latitude

    Output:
        Nx2 numpy array:
        [x_meters, y_meters]
    """

    lon = df['x'].to_numpy(dtype=float)
    lat = df['y'].to_numpy(dtype=float)

    # Use the mean latitude of the current dataset/window.
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)

    earth_radius = 6371000.0

    # Equirectangular/local tangent-plane approximation.
    x_m = (
        np.radians(lon - lon0)
        * earth_radius
        * np.cos(np.radians(lat0))
    )

    y_m = (
        np.radians(lat - lat0)
        * earth_radius
    )

    return np.column_stack((x_m, y_m))


# ============================================================
# 2. Douglas-Peucker Simplification
# ============================================================

def perpendicular_distance(point, start, end):
    """
    Calculate the perpendicular distance from a point to a line
    segment.

    All coordinates must be in meters.
    """

    if np.allclose(start, end):
        return np.linalg.norm(point - start)

    line_vec = end - start
    point_vec = point - start

    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq == 0:
        return np.linalg.norm(point - start)

    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0.0, 1.0)

    projection = start + t * line_vec

    return np.linalg.norm(point - projection)


def douglas_peucker_with_tolerance(points, times, delta):
    """
    Douglas-Peucker trajectory simplification.

    Parameters
    ----------
    points : Nx2 numpy array
        Coordinates in METERS.

    times : array
        Timestamp corresponding to each point.

    delta : float
        Simplification tolerance in METERS.

    Returns
    -------
    segments :
        List of Nx2 endpoint arrays:
        [[start_point, end_point], ...]

    time_spans :
        List of:
        [(start_time, end_time), ...]

    tolerances :
        Maximum approximation error for each simplified segment.
    """

    n = len(points)

    if n < 2:
        return [], [], []

    if n == 2:
        segment = np.array([points[0], points[1]])

        return (
            [segment],
            [(times[0], times[1])],
            [0.0]
        )

    start = points[0]
    end = points[-1]

    max_dist = -1.0
    max_index = -1

    for i in range(1, n - 1):

        dist = perpendicular_distance(
            points[i],
            start,
            end
        )

        if dist > max_dist:
            max_dist = dist
            max_index = i

    # --------------------------------------------------------
    # Split trajectory if maximum error exceeds tolerance.
    # --------------------------------------------------------

    if max_dist > delta:

        left_segments, left_times, left_tols = (
            douglas_peucker_with_tolerance(
                points[:max_index + 1],
                times[:max_index + 1],
                delta
            )
        )

        right_segments, right_times, right_tols = (
            douglas_peucker_with_tolerance(
                points[max_index:],
                times[max_index:],
                delta
            )
        )

        return (
            left_segments + right_segments,
            left_times + right_times,
            left_tols + right_tols
        )

    # --------------------------------------------------------
    # Entire trajectory section can be represented by one
    # simplified segment.
    # --------------------------------------------------------

    segment = np.array([start, end])

    return (
        [segment],
        [(times[0], times[-1])],
        [max(0.0, max_dist)]
    )


# ============================================================
# 3. True Line Segment Distance
# ============================================================

def cross_2d(a, b):
    """
    2D scalar cross product.
    """
    return a[0] * b[1] - a[1] * b[0]


def segments_intersect(A, B, C, D):
    """
    Determine whether line segments AB and CD intersect.

    Coordinates are in meters.
    """

    r = B - A
    s = D - C

    r_cross_s = cross_2d(r, s)
    c_minus_a = C - A

    q_cross_r = cross_2d(c_minus_a, r)

    # Parallel segments
    if np.isclose(r_cross_s, 0.0):

        # Non-collinear
        if not np.isclose(q_cross_r, 0.0):
            return False

        # Collinear - check bounding boxes
        min_ax = min(A[0], B[0])
        max_ax = max(A[0], B[0])

        min_ay = min(A[1], B[1])
        max_ay = max(A[1], B[1])

        min_cx = min(C[0], D[0])
        max_cx = max(C[0], D[0])

        min_cy = min(C[1], D[1])
        max_cy = max(C[1], D[1])

        overlap_x = (
            max(min_ax, min_cx)
            <=
            min(max_ax, max_cx)
        )

        overlap_y = (
            max(min_ay, min_cy)
            <=
            min(max_ay, max_cy)
        )

        return overlap_x and overlap_y

    t = cross_2d(c_minus_a, s) / r_cross_s
    u = cross_2d(c_minus_a, r) / r_cross_s

    return (
        0.0 <= t <= 1.0
        and
        0.0 <= u <= 1.0
    )


def point_to_segment_distance(P, A, B):
    """
    Minimum Euclidean distance between point P and segment AB.
    """

    AB = B - A
    AB_sq = np.dot(AB, AB)

    if AB_sq == 0:
        return np.linalg.norm(P - A)

    t = np.dot(P - A, AB) / AB_sq
    t = np.clip(t, 0.0, 1.0)

    projection = A + t * AB

    return np.linalg.norm(P - projection)


def segment_to_segment_distance(A, B, C, D):
    """
    True minimum distance between line segments AB and CD.

    This correctly handles:

        1. Intersecting segments
        2. Parallel segments
        3. Crossing segments
        4. Endpoint-to-segment cases

    Coordinates are in meters.
    """

    # If the segments intersect, distance is exactly zero.
    if segments_intersect(A, B, C, D):
        return 0.0

    d1 = point_to_segment_distance(A, C, D)
    d2 = point_to_segment_distance(B, C, D)
    d3 = point_to_segment_distance(C, A, B)
    d4 = point_to_segment_distance(D, A, B)

    return min(d1, d2, d3, d4)


# ============================================================
# 4. Temporal Overlap
# ============================================================

def segments_temporally_overlap(
    start1,
    end1,
    start2,
    end2
):
    """
    Determine whether two simplified trajectory segments
    overlap in time.
    """

    return (
        start1 <= end2
        and
        end1 >= start2
    )


# ============================================================
# 5. CuTS Trajectory DBSCAN
# ============================================================

def traj_dbscan(segments, eps, min_samples):
    """
    CuTS trajectory clustering using an R-tree spatial index.

    Each trajectory segment is indexed by its bounding box.
    Candidate pairs are then filtered by:
        1. Temporal overlap
        2. Expanded bounding-box intersection
        3. Exact segment-to-segment distance

    Finally, connected trajectory segments are clustered using
    DBSCAN-style expansion.

    Expected segment format:

        (
            segment_coordinates,
            (start_time, end_time),
            tolerance_meters,
            object_id
        )

    where segment_coordinates is:

        np.array([
            [x1, y1],
            [x2, y2]
        ])
    """

    from rtree import index

    n = len(segments)

    if n == 0:
        return []

    print(
        f"    Clustering {n:,} trajectory segments..."
    )

    # ==========================================================
    # 1. Extract segment information
    # ==========================================================

    starts = np.array(
        [s[0][0] for s in segments],
        dtype=float
    )

    ends = np.array(
        [s[0][1] for s in segments],
        dtype=float
    )

    tolerances = np.array(
        [s[2] for s in segments],
        dtype=float
    )

    object_ids = [
        s[3]
        for s in segments
    ]

    # ==========================================================
    # 2. Convert timestamps to pandas Timestamps
    # ==========================================================

    time_starts = np.array(
        [
            pd.Timestamp(s[1][0]).value
            for s in segments
        ],
        dtype=np.int64
    )

    time_ends = np.array(
        [
            pd.Timestamp(s[1][1]).value
            for s in segments
        ],
        dtype=np.int64
    )

    # ==========================================================
    # 3. Convert longitude/latitude to meters
    #
    # Beijing is approximately 39.9 degrees latitude.
    #
    # x = longitude
    # y = latitude
    # ==========================================================

    kx = (
        111000.0
        * np.cos(np.radians(39.9))
    )

    ky = 111000.0

    starts_m = starts * np.array([kx, ky])
    ends_m = ends * np.array([kx, ky])

    # ==========================================================
    # 4. Calculate segment bounding boxes
    # ==========================================================

    min_x = np.minimum(
        starts_m[:, 0],
        ends_m[:, 0]
    )

    max_x = np.maximum(
        starts_m[:, 0],
        ends_m[:, 0]
    )

    min_y = np.minimum(
        starts_m[:, 1],
        ends_m[:, 1]
    )

    max_y = np.maximum(
        starts_m[:, 1],
        ends_m[:, 1]
    )

    # ==========================================================
    # 5. Build R-tree
    # ==========================================================

    print("    Building R-tree...")

    rtree_index = index.Index()

    for i in range(n):

        rtree_index.insert(
            i,
            (
                min_x[i],
                min_y[i],
                max_x[i],
                max_y[i]
            )
        )

    print(
        f"    R-tree contains {n:,} segments"
    )

    # ==========================================================
    # 6. Build adjacency graph
    # ==========================================================

    adjacency = [
        set()
        for _ in range(n)
    ]

    candidate_pairs = 0
    valid_pairs = 0

    # ==========================================================
    # 7. Search each segment against the R-tree
    # ==========================================================

    for i in range(n):

        required_radius_base = (
            eps + tolerances[i]
        )

        # Expand this segment's bounding box.
        #
        # We use the maximum possible tolerance for the
        # candidate search. Exact tolerance is checked later.
        max_tol = np.max(tolerances)

        search_distance = (
            eps
            + tolerances[i]
            + max_tol
        )

        query_box = (
            min_x[i] - search_distance,
            min_y[i] - search_distance,
            max_x[i] + search_distance,
            max_y[i] + search_distance
        )

        candidates = rtree_index.intersection(
            query_box
        )

        for j in candidates:

            # Avoid checking the same pair twice
            if j <= i:
                continue

            candidate_pairs += 1

            # ==================================================
            # 8. Temporal overlap
            # ==================================================

            if (
                time_starts[i] > time_ends[j]
                or
                time_starts[j] > time_ends[i]
            ):
                continue

            # ==================================================
            # 9. Exact required distance
            #
            # CuTS distance threshold:
            #
            # epsilon + tolerance_i + tolerance_j
            # ==================================================

            required_distance = (
                eps
                + tolerances[i]
                + tolerances[j]
            )

            # ==================================================
            # 10. Exact bounding-box pruning
            # ==================================================

            if (
                max_x[i] + required_distance
                < min_x[j]
                or
                min_x[i] - required_distance
                > max_x[j]
                or
                max_y[i] + required_distance
                < min_y[j]
                or
                min_y[i] - required_distance
                > max_y[j]
            ):
                continue

            # ==================================================
            # 11. Exact segment-to-segment distance
            # ==================================================

            distance = segment_to_segment_distance(
                starts_m[i],
                ends_m[i],
                starts_m[j],
                ends_m[j]
            )

            # ==================================================
            # 12. Determine whether segments are neighbors
            # ==================================================

            if distance <= required_distance:

                adjacency[i].add(j)
                adjacency[j].add(i)

                valid_pairs += 1

    # ==========================================================
    # 13. Print diagnostics
    # ==========================================================

    print(
        f"    R-tree candidate pairs: "
        f"{candidate_pairs:,}"
    )

    print(
        f"    Valid trajectory pairs: "
        f"{valid_pairs:,}"
    )

    # ==========================================================
    # 14. DBSCAN-style clustering
    # ==========================================================

    print("    Running DBSCAN...")

    labels = np.full(
        n,
        -1,
        dtype=int
    )

    visited = np.zeros(
        n,
        dtype=bool
    )

    cluster_id = 0

    for i in range(n):

        if visited[i]:
            continue

        visited[i] = True

        neighbors = list(
            adjacency[i]
        )

        # Include the point itself when checking
        # min_samples.
        if len(neighbors) + 1 < min_samples:
            continue

        labels[i] = cluster_id

        seed_set = set(
            neighbors
        )

        seed_list = list(
            seed_set
        )

        position = 0

        while position < len(seed_list):

            current = seed_list[position]
            position += 1

            if not visited[current]:

                visited[current] = True

                current_neighbors = list(
                    adjacency[current]
                )

                if (
                    len(current_neighbors) + 1
                    >= min_samples
                ):

                    for neighbor in current_neighbors:

                        if neighbor not in seed_set:

                            seed_set.add(
                                neighbor
                            )

                            seed_list.append(
                                neighbor
                            )

            if labels[current] == -1:
                labels[current] = cluster_id

        cluster_id += 1

    # ==========================================================
    # 15. Convert clusters to CuTS format
    # ==========================================================

    clusters = []

    for label in range(cluster_id):

        indices = np.where(
            labels == label
        )[0]

        if len(indices) == 0:
            continue

        cluster_objects = {
            object_ids[idx]
            for idx in indices
        }

        # CuTS requires at least m distinct objects.
        if len(cluster_objects) >= min_samples:

            clusters.append({
                'objects': cluster_objects,
                'assigned': False
            })

    print(
        f"    Trajectory clusters: "
        f"{len(clusters):,}"
    )

    return clusters

# ============================================================
# 6. CuTS Filter - Algorithm 2
# ============================================================

def cuts_filter(
    df,
    m,
    k,
    e_meters,
    delta,
    lambda_time
):
    """
    CuTS Filter Step.

    Parameters
    ----------
    df :
        DataFrame containing:

            id
            time
            x = longitude
            y = latitude

    m :
        Minimum number of objects.

    k :
        Number of lambda-time partitions required.

    e_meters :
        Convoy distance threshold.

    delta :
        Douglas-Peucker tolerance in meters.

    lambda_time :
        Time partition length in minutes.
    """

    print("\n" + "=" * 60)
    print("CuTS FILTER")
    print("=" * 60)

    # --------------------------------------------------------
    # Ensure data is sorted exactly the same way as CMC.
    # --------------------------------------------------------

    df = df.sort_values(
        by=['id', 'time']
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Convert all coordinates to meters.
    #
    # This means:
    #
    #     delta = meters
    #
    # rather than degrees.
    # --------------------------------------------------------

    metric_coordinates = create_metric_coordinates(df)

    df_metric = df.copy()

    df_metric['x_m'] = metric_coordinates[:, 0]
    df_metric['y_m'] = metric_coordinates[:, 1]

    # --------------------------------------------------------
    # Douglas-Peucker simplification.
    # --------------------------------------------------------

    simplified_trajectories = {}

    total_segments = 0

    print(
        f"Simplifying trajectories with "
        f"delta = {delta:.2f} meters..."
    )

    for obj_id, group in df_metric.groupby('id'):

        group = group.sort_values(
            'time'
        ).reset_index(drop=True)

        if len(group) < 2:
            continue

        points = group[
            ['x_m', 'y_m']
        ].to_numpy(dtype=float)

        times = group[
            'time'
        ].to_numpy()

        (
            segs,
            time_spans,
            tols
        ) = douglas_peucker_with_tolerance(
            points,
            times,
            delta
        )

        trajectory_segments = []

        for seg, time_span, tol in zip(
            segs,
            time_spans,
            tols
        ):

            trajectory_segments.append(
                (
                    seg,
                    time_span,
                    tol,
                    obj_id
                )
            )

        simplified_trajectories[obj_id] = (
            trajectory_segments
        )

        total_segments += len(
            trajectory_segments
        )

    print(
        f"Original trajectory points : {len(df):,}"
    )

    print(
        f"Simplified segments         : {total_segments:,}"
    )

    # --------------------------------------------------------
    # Time domain.
    # --------------------------------------------------------

    min_time = df['time'].min()
    max_time = df['time'].max()

    # Normalize the starting point to the requested
    # lambda-minute partition.
    current_time = min_time

    V = []
    V_candidates = []

    partition_number = 0

    # --------------------------------------------------------
    # Process lambda-sized temporal partitions.
    # --------------------------------------------------------

    while current_time < max_time:

        partition_number += 1

        next_time = (
            current_time
            + pd.Timedelta(
                minutes=lambda_time
            )
        )

        print(
            f"Processing partition "
            f"{partition_number}: "
            f"{current_time} -> {next_time}"
        )

        V_next = []

        # ----------------------------------------------------
        # Construct G.
        #
        # A simplified segment is inserted into every
        # partition that it intersects.
        # ----------------------------------------------------

        G = []

        for obj_id, trajectory in simplified_trajectories.items():

            for (
                seg,
                (t_start, t_end),
                tol,
                segment_obj_id
            ) in trajectory:

                if (
                    t_start <= next_time
                    and
                    t_end >= current_time
                ):

                    G.append(
                        (
                            seg,
                            (t_start, t_end),
                            tol,
                            segment_obj_id
                        )
                    )

        print(
            f"    Segments in temporal window: "
            f"{len(G):,}"
        )

        # ----------------------------------------------------
        # Cluster trajectory segments.
        # ----------------------------------------------------

        C = traj_dbscan(
            G,
            e_meters,
            m
        )

        # ----------------------------------------------------
        # Continue existing candidate convoys.
        # ----------------------------------------------------

        for v in V:

            v['assigned'] = False

            for c in C:

                intersect = (
                    c['objects']
                    .intersection(
                        v['objects']
                    )
                )

                if len(intersect) >= m:

                    v['assigned'] = True
                    c['assigned'] = True

                    new_v = {
                        'objects': intersect,
                        'start_time': v[
                            'start_time'
                        ],
                        'end_time': next_time,
                        'lifetime': (
                            v['lifetime']
                            + lambda_time
                        ),
                        'assigned': False
                    }

                    V_next.append(new_v)

            # ------------------------------------------------
            # Candidate ended.
            # ------------------------------------------------

            if (
                not v['assigned']
                and
                (
                    v['lifetime']
                    / lambda_time
                ) >= k
            ):

                V_candidates.append(v)

        # ----------------------------------------------------
        # Start new candidate convoys.
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Final active candidates.
    # --------------------------------------------------------

    for v in V:

        if (
            v['lifetime']
            / lambda_time
        ) >= k:

            V_candidates.append(v)

    # --------------------------------------------------------
    # Remove exact duplicate candidates.
    # --------------------------------------------------------

    unique_candidates = []

    seen = set()

    for v in V_candidates:

        key = (
            frozenset(v['objects']),
            v['start_time'],
            v['end_time']
        )

        if key not in seen:

            seen.add(key)
            unique_candidates.append(v)

    print(
        "\nCuTS Filter Candidates: "
        f"{len(unique_candidates)}"
    )

    return unique_candidates


# ============================================================
# 7. Candidate Coverage Diagnostics
# ============================================================

def compare_cmc_to_cuts_candidates(
    cmc_convoys,
    cuts_candidates
):
    """
    Determine whether the CuTS filter is potentially
    eliminating CMC-discovered convoys.

    A CMC convoy is considered covered when a CuTS candidate:

        1. Contains all CMC objects.
        2. Has overlapping time.
    """

    print("\n" + "=" * 60)
    print("CMC vs CuTS FILTER COVERAGE")
    print("=" * 60)

    if len(cmc_convoys) == 0:

        print(
            "CMC discovered zero convoys."
        )

        return

    covered = 0
    missing = []

    for i, convoy in enumerate(cmc_convoys):

        cmc_objects = set(
            convoy['objects']
        )

        cmc_start = convoy[
            'start_time'
        ]

        cmc_end = convoy[
            'end_time'
        ]

        found = False

        for candidate in cuts_candidates:

            candidate_objects = set(
                candidate['objects']
            )

            candidate_start = candidate[
                'start_time'
            ]

            candidate_end = candidate[
                'end_time'
            ]

            object_match = (
                cmc_objects
                .issubset(
                    candidate_objects
                )
            )

            time_match = (
                candidate_start <= cmc_end
                and
                candidate_end >= cmc_start
            )

            if object_match and time_match:

                found = True
                break

        if found:

            covered += 1

        else:

            missing.append(
                {
                    'index': i + 1,
                    'objects': cmc_objects,
                    'start_time': cmc_start,
                    'end_time': cmc_end
                }
            )

    print(
        f"CMC convoys                  : "
        f"{len(cmc_convoys)}"
    )

    print(
        f"Covered by CuTS candidates   : "
        f"{covered}"
    )

    print(
        f"NOT covered by CuTS filter   : "
        f"{len(missing)}"
    )

    if missing:

        print(
            "\nCMC convoys potentially "
            "eliminated by CuTS:"
        )

        for item in missing:

            print(
                f"  CMC #{item['index']}: "
                f"Objects={item['objects']} | "
                f"{item['start_time']} -> "
                f"{item['end_time']}"
            )

    print("=" * 60)


# ============================================================
# 8. CuTS Refinement - Algorithm 3
# ============================================================

def cuts_refinement(
    df,
    candidates,
    m,
    k_cmc,
    e_meters,
    lambda_time=10
):
    """
    CuTS Refinement Step.

    Runs the CMC algorithm on the ORIGINAL trajectory
    points associated with each CuTS candidate.
    """

    print("\n" + "=" * 60)
    print("CuTS REFINEMENT")
    print("=" * 60)

    final_convoys = []

    for candidate_number, v in enumerate(
        candidates,
        start=1
    ):

        candidate_objects = set(
            v['objects']
        )

        candidate_start = v[
            'start_time'
        ]

        candidate_end = v[
            'end_time'
        ]

        print(
            f"Refining candidate "
            f"{candidate_number}/"
            f"{len(candidates)}: "
            f"{candidate_objects} | "
            f"{candidate_start} -> "
            f"{candidate_end}"
        )

        # ----------------------------------------------------
        # Small temporal buffer.
        #
        # This prevents a candidate boundary from clipping
        # a CMC convoy.
        # ----------------------------------------------------

        buffer = pd.Timedelta(
            minutes=lambda_time
        )

        t_start = candidate_start - buffer
        t_end = candidate_end + buffer

        df_subset = df[
            (
                df['id'].isin(
                    candidate_objects
                )
            )
            &
            (
                df['time'] >= t_start
            )
            &
            (
                df['time'] <= t_end
            )
        ].copy()

        if len(df_subset) == 0:
            continue

        # ----------------------------------------------------
        # Make sure refinement receives the same data format
        # as the standalone CMC program.
        # ----------------------------------------------------

        df_subset = df_subset.sort_values(
            by=['id', 'time']
        ).reset_index(drop=True)

        refined_convoys = (
            cmc.discover_convoys_cmc(
                df_subset,
                m,
                k_cmc,
                e_meters
            )
        )

        final_convoys.extend(
            refined_convoys
        )

    # --------------------------------------------------------
    # Deduplicate refined convoys.
    # --------------------------------------------------------

    unique_convoys = []

    seen = set()

    for convoy in final_convoys:

        key = (
            frozenset(
                convoy['objects']
            ),
            convoy['start_time'],
            convoy['end_time']
        )

        if key not in seen:

            seen.add(key)
            unique_convoys.append(
                convoy
            )

    print(
        f"\nRefinement produced "
        f"{len(unique_convoys)} unique convoys."
    )

    return unique_convoys


# ============================================================
# 9. Main CuTS Algorithm
# ============================================================

def discover_convoys_cuts(
    df,
    m,
    k_cuts,
    k_cmc,
    e_meters,
    delta,
    lambda_time,
    cmc_baseline=None
):
    """
    Complete CuTS algorithm:

        Algorithm 2: Filter
        Algorithm 3: Refinement

    cmc_baseline is optional and is only used for diagnostics.
    """

    total_start = time.time()

    # ========================================================
    # FILTER
    # ========================================================

    print("\n")
    print("=" * 60)
    print("RUNNING CuTS FILTER")
    print("=" * 60)

    filter_start = time.time()

    candidates = cuts_filter(
        df=df,
        m=m,
        k=k_cuts,
        e_meters=e_meters,
        delta=delta,
        lambda_time=lambda_time
    )

    filter_end = time.time()

    filter_duration = (
        filter_end
        - filter_start
    )

    print(
        f"\nFilter execution time: "
        f"{filter_duration:.2f} seconds"
    )

    print(
        f"Candidates found: "
        f"{len(candidates)}"
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if cmc_baseline is not None:

        compare_cmc_to_cuts_candidates(
            cmc_baseline,
            candidates
        )

    # ========================================================
    # REFINEMENT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("RUNNING CuTS REFINEMENT")
    print("=" * 60)

    refinement_start = time.time()

    actual_convoys = cuts_refinement(
        df=df,
        candidates=candidates,
        m=m,
        k_cmc=k_cmc,
        e_meters=e_meters,
        lambda_time=lambda_time
    )

    refinement_end = time.time()

    refinement_duration = (
        refinement_end
        - refinement_start
    )

    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_end = time.time()

    total_duration = (
        total_end
        - total_start
    )

    print("\n" + "=" * 60)
    print("CuTS TIMING BREAKDOWN")
    print("=" * 60)

    print(
        f"Filter Time     : "
        f"{filter_duration:.2f} seconds "
        f"({filter_duration / total_duration * 100:.1f}%)"
    )

    print(
        f"Refinement Time : "
        f"{refinement_duration:.2f} seconds "
        f"({refinement_duration / total_duration * 100:.1f}%)"
    )

    print(
        f"Total Time      : "
        f"{total_duration:.2f} seconds"
    )

    print("=" * 60)

    return actual_convoys, candidates


# ============================================================
# 10. Main Program
# ============================================================

if __name__ == "__main__":

    file = "tdrive_processed.csv"

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("Loading data...")

    my_data = pd.read_csv(
        file,
        header=0,
        parse_dates=['time']
    )

    print(
        f"Data loaded. "
        f"Total rows: {len(my_data):,}"
    )

    # ========================================================
    # EXACT SAME TIME WINDOW AS CMC
    # ========================================================

    start_time = pd.Timestamp(
        "2008-02-02 13:00:00"
    )

    end_time = pd.Timestamp(
        "2008-02-02 18:00:00"
    )

    windowed_data = my_data[
        (my_data['time'] >= start_time)
        &
        (my_data['time'] <= end_time)
    ].copy()

    # ========================================================
    # SORT EXACTLY LIKE CMC
    # ========================================================

    windowed_data = (
        windowed_data
        .sort_values(
            by=['id', 'time']
        )
        .reset_index(drop=True)
    )

    print(
        f"\nData filtered from "
        f"{start_time} "
        f"to "
        f"{end_time}"
    )

    print(
        f"Windowed data points: "
        f"{len(windowed_data):,}"
    )

    print(
        f"Unique taxis: "
        f"{windowed_data['id'].nunique():,}"
    )

    # ========================================================
    # PARAMETERS
    # ========================================================

    m_objects = 3

    # CuTS filter lifetime.
    # 2 x 10-minute partitions = 20 minutes.
    k_cuts = 2

    # CMC refinement lifetime.
    k_cmc = 3

    # Convoy spatial threshold.
    epsilon = 100.0

    # Douglas-Peucker simplification tolerance.
    simplification_tolerance = 100.0

    # CuTS temporal partition size.
    time_window = 10

    print("\n")
    print("=" * 60)
    print("PARAMETERS")
    print("=" * 60)

    print(
        f"m                 = {m_objects}"
    )

    print(
        f"k_cuts            = {k_cuts}"
    )

    print(
        f"k_cmc             = {k_cmc}"
    )

    print(
        f"epsilon           = {epsilon} m"
    )

    print(
        f"delta             = "
        f"{simplification_tolerance} m"
    )

    print(
        f"lambda            = "
        f"{time_window} minutes"
    )

    print("=" * 60)

    # ========================================================
    # FIRST: RUN CMC BASELINE
    #
    # This is the exact same dataframe that CuTS receives.
    # ========================================================

    print("\n")
    print("=" * 60)
    print("RUNNING CMC BASELINE")
    print("=" * 60)

    cmc_start = time.time()

    cmc_convoys = cmc.discover_convoys_cmc(
        windowed_data,
        m=m_objects,
        k=k_cmc,
        e_meters=epsilon
    )

    cmc_end = time.time()

    print(
        f"\nCMC execution time: "
        f"{cmc_end - cmc_start:.2f} seconds"
    )

    print(
        f"CMC convoys discovered: "
        f"{len(cmc_convoys)}"
    )

    for i, convoy in enumerate(
        cmc_convoys,
        start=1
    ):

        print(
            f"CMC Convoy {i}: "
            f"Objects={convoy['objects']} | "
            f"Time="
            f"{convoy['start_time']} -> "
            f"{convoy['end_time']}"
        )

    # ========================================================
    # RUN CuTS
    #
    # IMPORTANT:
    # Same windowed_data.
    # ========================================================

    print("\n")
    print("=" * 60)
    print("RUNNING CuTS")
    print("=" * 60)

    cuts_convoys, cuts_candidates = (
        discover_convoys_cuts(
            df=windowed_data,
            m=m_objects,
            k_cuts=k_cuts,
            k_cmc=k_cmc,
            e_meters=epsilon,
            delta=simplification_tolerance,
            lambda_time=time_window,
            cmc_baseline=cmc_convoys
        )
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"CMC convoys       : "
        f"{len(cmc_convoys)}"
    )

    print(
        f"CuTS candidates   : "
        f"{len(cuts_candidates)}"
    )

    print(
        f"CuTS final convoys: "
        f"{len(cuts_convoys)}"
    )

    print("=" * 60)

    # ========================================================
    # PRINT CuTS CONVOYS
    # ========================================================

    for i, convoy in enumerate(
        cuts_convoys,
        start=1
    ):

        print(
            f"CuTS Convoy {i}: "
            f"Objects={convoy['objects']} | "
            f"Time="
            f"{convoy['start_time']} -> "
            f"{convoy['end_time']}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if cuts_convoys:

        df_convoys = pd.DataFrame(
            cuts_convoys
        )

        output_file = (
            f"cuts_convoys_"
            f"{m_objects}_"
            f"{k_cmc}_"
            f"{epsilon}.csv"
        )

        df_convoys.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nSaved CuTS results to: "
            f"{output_file}"
        )

        # ====================================================
        # VISUALIZE
        # ====================================================

        viz.visualize_convoys(
            windowed_data,
            cuts_convoys,
            file=(
                f"cuts_convoys_map_"
                f"{m_objects}_"
                f"{k_cmc}_"
                f"{epsilon}"
            )
        )

    else:

        print(
            "\nNo CuTS convoys discovered; "
            "skipping visualization."
        )