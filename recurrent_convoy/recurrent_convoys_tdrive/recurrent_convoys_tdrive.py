"""Recurrent convoys on T-drive; Yadamjav et al., ACM TIST 2020.

Updated for generate_synthetic_beijing_day_routes.py.
Only trajectories.csv is used for detection; event labels are never inputs. This is a research reimplementation, not the authors' code or
an experimentally verified reproduction of their counts/timings.

Like cuts_timed_classes_v3.py: editable parameters, named pipeline stages,
typed records, Tee logging, timing summaries, and CSV exports. No CuTS/DP step:
the recurrent-convoy paper clusters timestamp snapshots, not simplified lines.

Python >= 3.10. pip install numpy pandas scipy scikit-learn
Optional: pip install rtree (paper's temporal/cardinality R-tree).
Generator-compatible defaults: observed 1-second grid, tau=4, k=3 samples
(2 elapsed seconds), rho=7 days, entire input query window, no overlap.
Sparse observations are NOT filled or treated as consecutive seconds.
CSV input must be time-sorted when --stream-input is enabled (the generator is).
Use --no-stream-input for unsorted CSV or explicit interpolation.

Run: python recurrent_convoys_tdrive.py --input synthetic_beijing_full_routes
Hourly only: add --rho-seconds 86400
Duration: --min-duration-seconds 8 means 9 samples at one-second cadence.
--k remains a sample count for compatibility; do not supply both duration flags.
All result files keep their previous schemas. Geometry is in projected meters.
Streaming bounds input-frame memory, not stored clusters or result-graph size.
Full six-month performance depends on noise and output size; try a short run first.


python recurrent_convoys_tdrive.py --input seed11_every_second/trajectories.csv --output synthetic_results_4hr --rho-seconds 14400 --step-seconds 1 --sampling observed
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import json
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


# ============================================================
# Editable experiment parameters (also overridable on command line)
# ============================================================
@dataclass
class Parameters:
    input: str = "synthetic_beijing_full_routes"  # id,time,x,y; x=longitude, y=latitude
    output: str = "recurrent_convoy_results"
    tau: int = 4                        # common objects throughout ONE occurrence
    k: int = 3                          # consecutive samples; NOT minutes
    rho_seconds: float = 604800          # START-to-START recurrence limit
    epsilon_m: float = 100.0            # DBSCAN radius
    min_pts: int = 4                    # DBSCAN includes the point itself
    tau_min: int = 4                    # minimum supported query prominence
    delta_m: float = 100.0              # cluster Hausdorff similarity threshold
    metric: str = "hd"                 # hd or tau_hd (documented interpretation)
    step_seconds: int = 1               # expected sampling interval
    sampling: str = "observed"         # observed or interpolate; NEVER silently bin
    max_gap_seconds: float = 300.0      # interpolation only
    coordinates: str = "lonlat"        # lonlat or meters
    history_start: str | None = None    # retain history BEFORE the query window
    query_end: str | None = None        # default final observation on grid
    window_seconds: float = 0.0         # 0 = entire input; otherwise inclusive window
    engine: str = "convoy"             # baseline, lookup, intersection, convoy
    index_backend: str = "auto"        # auto, sorted, rtree
    occurrence_policy: str = "paper"   # paper Alg. 1 or all (all valid sequences)
    mbr_pruning: bool = True            # Section 4.4.1; safe for both metrics
    allow_overlap: bool = False          # Def. 3 has no non-overlap constraint
    max_generated_rows: int = 5_000_000 # fail explicitly; never silently truncate

    min_duration_seconds: float | None = None  # elapsed duration, overrides k
    stream_input: bool = True           # sorted CSV, bounded input chunks
    chunk_rows: int = 250_000
    similarity_cache_limit: int = 100_000

    def validate(self):
        if self.min_duration_seconds is not None:
            if not math.isfinite(self.min_duration_seconds) or self.min_duration_seconds <= 0:
                raise ValueError("min_duration_seconds must be finite and positive")
            if self.step_seconds < 1:
                raise ValueError("step_seconds must be positive")
            self.k = max(2, math.ceil(self.min_duration_seconds / self.step_seconds) + 1)
        if self.chunk_rows < 1 or self.similarity_cache_limit < 1:
            raise ValueError("chunk_rows and similarity_cache_limit must be positive")
        if not (self.tau >= self.tau_min >= 1 and self.k >= 2 and self.min_pts >= 1):
            raise ValueError("Require tau >= tau_min >= 1, k >= 2, min_pts >= 1")
        if self.step_seconds < 1 or self.epsilon_m <= 0 or self.delta_m < 0:
            raise ValueError("Require step_seconds >= 1, epsilon_m > 0, delta_m >= 0")
        if self.rho_seconds <= 0 or (self.window_seconds != 0 and self.window_seconds < self.k * self.step_seconds):
            raise ValueError("rho must be positive; window must be 0 (entire input) or cover at least k samples")
        if self.max_gap_seconds <= 0 or self.max_generated_rows < 1:
            raise ValueError("max_gap_seconds and max_generated_rows must be positive")
        for value, choices in [(self.metric, ('hd', 'tau_hd')),
                               (self.sampling, ('observed', 'interpolate')),
                               (self.coordinates, ('lonlat', 'meters')),
                               (self.engine, ('baseline', 'lookup', 'intersection', 'convoy')),
                               (self.index_backend, ('auto', 'sorted', 'rtree')),
                               (self.occurrence_policy, ('paper', 'all'))]:
            if value not in choices:
                raise ValueError(f"Invalid option {value!r}; choose {choices}")


DEFAULTS = Parameters()


# ============================================================
# Type definitions and terminal logging
# ============================================================
@dataclass
class Cluster:
    cid: int
    tick: int
    objects: frozenset[str]
    points: np.ndarray
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Convoy:
    clusters: tuple[int, ...]
    objects: frozenset[str]
    start: int
    end: int


@dataclass
class ClusterIndex:
    clusters: dict[int, Cluster]
    by_time: dict[int, list[int]]
    predecessors: dict[int, dict[int, frozenset[str]]] = field(default_factory=dict)
    chain_id: dict[int, int] = field(default_factory=dict)
    # For each cluster/object: final tick of its CONTINUOUS run in this index chain.
    run_end: dict[int, dict[str, int]] = field(default_factory=dict)
    chain_next: dict[int, int] = field(default_factory=dict)
    backend: str = "sorted"
    tree: object = None

    def range_clusters(self, first: int, last: int, tau: int):
        if self.tree is not None:
            ids = list(self.tree.intersection((first, tau, last, float('inf'))))
            return sorted(ids, key=lambda cid: (self.clusters[cid].tick, cid))
        times = sorted(self.by_time)
        lo, hi = bisect.bisect_left(times, first), bisect.bisect_right(times, last)
        return [cid for t in times[lo:hi] for cid in self.by_time[t]
                if len(self.clusters[cid].objects) >= tau]


class Tee:
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


def bbox(points):
    return (float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()))


def mbr_distance(a, b):
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0),
                      max(a[1] - b[3], b[1] - a[3], 0))


def tick_time(tick, p):
    return pd.Timestamp(tick * p.step_seconds, unit='s').isoformat()


# ============================================================
# Load CSV or original T-drive directory; project and sample
# ============================================================
def load_dataset(p):
    path = Path(p.input)
    if path.is_dir() and (path / 'trajectories.csv').is_file():
        path = path / 'trajectories.csv'
    if path.is_dir():
        files = sorted(path.rglob('*.txt'))
        if not files:
            raise ValueError("Raw T-drive directory contains no .txt files")
        frames = [pd.read_csv(f, names=['id', 'time', 'x', 'y'],
                              dtype={'id': str}) for f in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, dtype={'id': str})
    if not {'id', 'time', 'x', 'y'}.issubset(df.columns):
        raise ValueError("Expected CSV columns id,time,x,y (header required)")
    df = df[['id', 'time', 'x', 'y']].copy()
    original = len(df)
    # Treat timezone-naive input as one consistent local time axis. Do not mix zones.
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    if getattr(df['time'].dt, 'tz', None) is not None:
        raise ValueError("Use timezone-naive timestamps in a single consistent timezone")
    for col in ['x', 'y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df = df[np.isfinite(df['x']) & np.isfinite(df['y'])]
    if p.coordinates == 'lonlat':
        df = df[df.x.between(-180, 180) & df.y.between(-90, 90)]
    print(f"Dropped invalid rows: {original - len(df):,}")
    conflicts = df.duplicated(['id', 'time'], keep=False)
    if conflicts.any():
        dup = df[conflicts].groupby(['id', 'time'])[['x', 'y']].nunique()
        if (dup > 1).any().any():
            raise ValueError("Conflicting coordinates for the same id/time; resolve before mining")
    df = df.drop_duplicates(['id', 'time']).sort_values(['id', 'time'])
    if df.empty:
        raise ValueError("No valid observations")
    end = pd.Timestamp(p.query_end) if p.query_end else df.time.max().floor(f'{p.step_seconds}s')
    start = pd.Timestamp(p.history_start) if p.history_start else df.time.min()
    if end < start:
        raise ValueError("query_end precedes history_start")
    # Keep one bracketing sample on either side for explicitly requested interpolation.
    if p.sampling == 'interpolate':
        gap = pd.Timedelta(seconds=p.max_gap_seconds)
        df = df[(df.time >= start - gap) & (df.time <= end + gap)].copy()
    else:
        df = df[(df.time >= start) & (df.time <= end)].copy()
    if df.empty:
        raise ValueError("No observations in selected history range")
    projection = {'coordinates': p.coordinates}
    if p.coordinates == 'lonlat':
        # Same local equirectangular projection style as the attached CuTS code.
        lon0, lat0 = float(df.x.mean()), float(df.y.mean())
        scale_x = 111320.0 * math.cos(math.radians(lat0))
        df['x'] = (df.x - lon0) * scale_x
        df['y'] = (df.y - lat0) * 111320.0
        projection.update(lon0=lon0, lat0=lat0, meters_per_degree_x=scale_x,
                          meters_per_degree_y=111320.0)
    unit = p.step_seconds * 1_000_000_000
    if end.value % unit:
        raise ValueError("query_end must lie on the epoch-aligned sampling grid")
    first_tick = math.ceil(start.value / unit)
    last_tick = end.value // unit
    if p.sampling == 'observed':
        ns = df.time.astype('datetime64[ns]').astype('int64')
        if (ns % unit != 0).any():
            raise ValueError("Observed timestamps are off-grid. Choose the correct step or "
                             "explicitly use --sampling interpolate; times are never rounded.")
        df['tick'] = ns // unit
        sampled = df[['id', 'tick', 'x', 'y']].copy()
    else:
        pieces, count = [], 0
        for oid, group in df.groupby('id', sort=True):
            ns = group.time.astype('datetime64[ns]').astype('int64').to_numpy()
            lo = max(first_tick, math.ceil(int(ns[0]) / unit))
            hi = min(last_tick, int(ns[-1]) // unit)
            if hi < lo:
                continue
            # Avoid huge temporary arrays before checking the guard.
            if hi - lo + 1 > p.max_generated_rows:
                raise MemoryError("Single trajectory grid exceeds max_generated_rows")
            ticks = np.arange(lo, hi + 1, dtype=np.int64)
            targets = ticks * unit
            right = np.searchsorted(ns, targets)
            clipped = np.minimum(right, len(ns) - 1)
            exact = ns[clipped] == targets
            interior = (right > 0) & (right < len(ns))
            left = np.maximum(right - 1, 0)
            valid_gap = ns[clipped] - ns[left] <= p.max_gap_seconds * 1e9
            keep = exact | (interior & valid_gap)
            ticks, targets = ticks[keep], targets[keep]
            count += len(ticks)
            if count > p.max_generated_rows:
                raise MemoryError("Interpolation exceeds max_generated_rows; reduce range or increase guard")
            # Relative times avoid precision loss from epoch nanoseconds.
            relative = (ns - ns[0]) / 1e9
            wanted = (targets - ns[0]) / 1e9
            pieces.append(pd.DataFrame({'id': oid, 'tick': ticks,
                'x': np.interp(wanted, relative, group.x.to_numpy()),
                'y': np.interp(wanted, relative, group.y.to_numpy())}))
        sampled = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
            columns=['id', 'tick', 'x', 'y'])
    sampled = sampled[sampled.tick.between(first_tick, last_tick)]
    if sampled.empty:
        raise ValueError("No samples on the selected grid")
    print(f"Sample rows: {len(sampled):,}; taxis: {sampled.id.nunique():,}; "
          f"occupied timestamps: {sampled.tick.nunique():,}")
    return sampled, int(first_tick), int(last_tick), projection


# ============================================================
# DBSCAN at each timestamp (Section 4.1)
# ============================================================
def generate_clusters(df, p):
    clusters, by_time, rows = {}, {}, []
    for number, (tick, group) in enumerate(df.groupby('tick', sort=True), 1):
        begin = time.perf_counter()
        group = group.sort_values('id')  # deterministic DBSCAN border assignment
        points = group[['x', 'y']].to_numpy(dtype=float)
        labels = (DBSCAN(eps=p.epsilon_m, min_samples=p.min_pts,
                        algorithm='kd_tree', n_jobs=1).fit_predict(points)
                  if len(group) >= max(p.min_pts, p.tau_min)
                  else np.full(len(group), -1))
        ids = group.id.to_numpy()
        current = []
        for label in sorted(set(labels) - {-1}):
            mask = labels == label
            if int(mask.sum()) < p.tau_min:
                continue
            cid = len(clusters)
            xy = points[mask].copy()
            clusters[cid] = Cluster(cid, int(tick), frozenset(map(str, ids[mask])), xy, bbox(xy))
            current.append(cid)
        if current:
            by_time[int(tick)] = current
        rows.append({'time': tick_time(int(tick), p), 'objects': len(group),
                     'clusters': len(current), 'seconds': time.perf_counter() - begin})
        if number % 1000 == 0:
            print(f"[Cluster] {number:,} occupied timestamps; {len(clusters):,} clusters")
    return ClusterIndex(clusters, by_time), rows


def stream_clusters(p):
    """Consume time-sorted mixed-cadence CSV, keeping complete timestamps together.

    Fixed Beijing projection matches the generator, so every chunk and every
    month uses the same coordinate system. No interpolation or forward fill.
    Input rows outside a requested history range are discarded early.
    """
    if p.sampling != 'observed':
        raise ValueError('Streaming requires observed sampling; use --no-stream-input for interpolation')
    path = Path(p.input)
    if path.is_dir():
        path = path / 'trajectories.csv'
    scale = 111320.0 * math.cos(math.radians(39.90))
    projection = {'coordinates': p.coordinates}
    if p.coordinates == 'lonlat':
        projection.update(lon0=116.40, lat0=39.90, meters_per_degree_x=scale,
                          meters_per_degree_y=111320.0)
    unit = p.step_seconds * 1_000_000_000
    def boundary(value):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None or stamp.value % unit:
            raise ValueError('Time boundaries must be timezone-naive and on the sampling grid')
        return stamp.value // unit
    lower = boundary(p.history_start) if p.history_start else None
    upper = boundary(p.query_end) if p.query_end else None
    if lower is not None and upper is not None and upper < lower:
        raise ValueError('query_end precedes history_start')
    index, statistics = ClusterIndex({}, {}), []
    carry, previous_ns, first, end = None, None, None, None
    observed_rows, dropped, duplicates = 0, 0, 0

    def consume(frame):
        nonlocal first, end, observed_rows, duplicates
        if frame.empty:
            return
        conflicts = frame.duplicated(['id', 'tick'], keep=False)
        if conflicts.any():
            counts = frame[conflicts].groupby(['id', 'tick'])[['x', 'y']].nunique()
            if (counts > 1).any().any():
                raise ValueError('Conflicting coordinates for the same id/time')
        n = len(frame)
        frame = frame.drop_duplicates(['id', 'tick'])
        duplicates += n - len(frame)
        observed_rows += len(frame)
        lo, hi = int(frame.tick.min()), int(frame.tick.max())
        first = lo if first is None else min(first, lo)
        end = hi if end is None else max(end, hi)
        part, rows = generate_clusters(frame, p)
        offset = len(index.clusters)
        for cid, c in part.clusters.items():
            c.cid = offset + cid
            index.clusters[c.cid] = c
        for tick, ids in part.by_time.items():
            index.by_time[tick] = [offset + cid for cid in ids]
        statistics.extend(rows)

    for chunk_number, frame in enumerate(pd.read_csv(path, dtype={'id': str},
            usecols=['id', 'time', 'x', 'y'], chunksize=p.chunk_rows), 1):
        original = len(frame)
        frame['time'] = pd.to_datetime(frame.time, errors='coerce')
        if getattr(frame.time.dt, 'tz', None) is not None:
            raise ValueError('Use timezone-naive timestamps')
        for col in ('x', 'y'):
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
        frame = frame.dropna()
        frame = frame[np.isfinite(frame.x) & np.isfinite(frame.y)]
        if p.coordinates == 'lonlat':
            frame = frame[frame.x.between(-180, 180) & frame.y.between(-90, 90)]
        dropped += original - len(frame)
        if frame.empty:
            continue
        ns = frame.time.astype('datetime64[ns]').astype('int64')
        if not ns.is_monotonic_increasing or (previous_ns is not None and int(ns.iloc[0]) < previous_ns):
            raise ValueError('CSV must be sorted by time; use --no-stream-input for unsorted input')
        previous_ns = int(ns.iloc[-1])
        if (ns % unit != 0).any():
            raise ValueError('Observed timestamps are off-grid; generator requires --step-seconds 1')
        frame['tick'] = ns // unit
        if lower is not None:
            frame = frame[frame.tick >= lower]
        if upper is not None:
            frame = frame[frame.tick <= upper]
        frame = frame[['id', 'tick', 'x', 'y']].copy()
        if p.coordinates == 'lonlat':
            frame['x'] = (frame.x - 116.40) * scale
            frame['y'] = (frame.y - 39.90) * 111320.0
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            continue
        last_tick = frame.tick.iloc[-1]
        carry = frame[frame.tick == last_tick].copy()
        consume(frame[frame.tick != last_tick])
        print(f'[Input] chunk {chunk_number}: {observed_rows:,} rows; {len(index.clusters):,} clusters', flush=True)
        if upper is not None and previous_ns // unit > upper:
            break
    if carry is not None:
        consume(carry)
    if first is None:
        raise ValueError('No valid observations in selected history range')
    print(f'Observed rows: {observed_rows:,}; invalid dropped: {dropped:,}; duplicates removed: {duplicates:,}')
    return index, statistics, int(lower if lower is not None else first), int(upper if upper is not None else end), projection


# ============================================================
# Intersection and distinctive convoy indexes (Sections 4.3/4.4)
# ============================================================
def build_index(index, p):
    cs = index.clusters
    previous, previous_tick = {}, None
    chains, active_common = {}, {}
    for tick, ids in sorted(index.by_time.items()):
        if previous_tick is None or tick != previous_tick + 1:
            previous = {}
        incoming, outgoing = {}, defaultdict(list)
        for cid in ids:
            overlaps = defaultdict(set)
            for oid in cs[cid].objects:
                if oid in previous:
                    overlaps[previous[oid]].add(oid)
            pred = {old: frozenset(objects) for old, objects in overlaps.items()
                    if len(objects) >= p.tau_min}
            index.predecessors[cid] = pred
            incoming[cid] = pred
            for old in pred:
                outgoing[old].append(cid)
        for cid in ids:
            pred = incoming[cid]
            old = next(iter(pred)) if len(pred) == 1 else None
            extend = old is not None and len(outgoing[old]) == 1
            common = active_common.get(old, frozenset()) & cs[cid].objects if extend else frozenset()
            # Split on convergence/divergence AND loss of tau_min persistent objects.
            if extend and len(common) >= p.tau_min:
                chain = index.chain_id[old]
                index.chain_next[old] = cid
            else:
                chain, common = len(chains), cs[cid].objects
                chains[chain] = []
            index.chain_id[cid] = chain
            active_common[cid] = common
            chains[chain].append(cid)
        previous = {oid: cid for cid in ids for oid in cs[cid].objects}
        previous_tick = tick
        active_common = {cid: active_common[cid] for cid in ids}
    # Offline finalization of index-chain object lifetimes. A missing/reappearing
    # object has separate runs, never a single first-to-last range.
    for chain in chains.values():
        next_ends = {}
        for cid in reversed(chain):
            current = {oid: next_ends.get(oid, cs[cid].tick) for oid in cs[cid].objects}
            index.run_end[cid] = current
            next_ends = current
    if p.index_backend != 'sorted':
        try:
            from rtree.index import Index, Property
            prop = Property()
            prop.index_capacity = prop.leaf_capacity = 100
            index.tree = Index(properties=prop)
            for cid, c in cs.items():
                index.tree.insert(cid, (c.tick, len(c.objects), c.tick, len(c.objects)))
            index.backend = 'rtree'
        except ImportError:
            if p.index_backend == 'rtree':
                raise ImportError("Install rtree or use --index-backend sorted") from None
    print(f"Index: {index.backend}; distinctive chains: {len(chains):,}; "
          f"intersection links: {sum(map(len, index.predecessors.values())):,}")
    return index


# ============================================================
# Convoy generation: Algorithm 1 and indexed variants
# ============================================================
def mine_convoys(index, p, first=None, last=None, stats=None):
    """Return completed candidates plus active candidates at the range boundary.

    paper policy: Algorithm 1 seeds only clusters extending no candidate.
    all policy: seed every cluster and retain every valid prefix; exponential
    output is possible. This implements the literal Def. 1 sequence universe.
    Engines change extension access, not candidate/occurrence semantics.
    """
    if stats is None:
        stats = Counter()
    cs = index.clusters
    if not cs:
        return []
    first = min(index.by_time) if first is None else first
    last = max(index.by_time) if last is None else last
    selected = defaultdict(list)
    for cid in index.range_clusters(first, last, p.tau):
        selected[cs[cid].tick].append(cid)
    active, output, previous_tick = [], {}, None

    def emit(g):
        if len(g.clusters) >= p.k:
            output[g.clusters] = g

    for tick, ids in sorted(selected.items()):
        if previous_tick is None or tick != previous_tick + 1:
            for g in active:
                emit(g)
            active = []
        lookup = {o: cid for cid in ids for o in cs[cid].objects}
        available = set(ids)
        new, extended_clusters = {}, set()
        for g in active:
            old = g.clusters[-1]
            if p.engine == 'baseline':
                candidates = ids
            elif p.engine == 'lookup':
                counts = Counter(lookup[o] for o in g.objects if o in lookup)
                candidates = [cid for cid, n in counts.items() if n >= p.tau]
            else:
                candidates = [cid for cid in ids
                              if len(index.predecessors.get(cid, {}).get(old, ())) >= p.tau]
            extended = False
            # Section 4.4 lifetime embedding avoids set intersection within an
            # index chain. Expired members MUST still be removed on each step.
            next_id = index.chain_next.get(old)
            for cid in candidates:
                if p.engine == 'convoy' and cid == next_id and cid in available:
                    ends = index.run_end[old]
                    common = frozenset(o for o in g.objects if ends[o] >= tick)
                    stats['lifespan_extensions'] += 1
                else:
                    common = g.objects & cs[cid].objects
                    stats['set_intersections'] += 1
                if len(common) < p.tau:
                    continue
                h = Convoy(g.clusters + (cid,), common, g.start, tick)
                new[h.clusters] = h
                extended_clusters.add(cid)
                extended = True
            if not extended:
                emit(g)
        seeds = ids if p.occurrence_policy == 'all' else [c for c in ids if c not in extended_clusters]
        for cid in seeds:
            new[(cid,)] = Convoy((cid,), cs[cid].objects, tick, tick)
        active = list(new.values())
        if p.occurrence_policy == 'all':
            for g in active:
                emit(g)
        stats['peak_active_candidates'] = max(stats['peak_active_candidates'], len(active))
        previous_tick = tick
    for g in active:  # Algorithm 2's omitted final flush is required for finite data.
        emit(g)
    return sorted(output.values(), key=lambda g: (g.start, g.end, g.clusters))


# ============================================================
# Similarity: full cluster geometry, shifted k-length subsequences
# ============================================================
class Similarity:
    def __init__(self, index, p):
        self.index, self.p = index, p
        self.cache = {}
        self.trees = {}
        self.stats = Counter()

    def cluster_similar(self, a, b):
        key = tuple(sorted((a, b)))
        if key in self.cache:
            self.stats['cluster_cache_hits'] += 1
            return self.cache[key]
        ca, cb = self.index.clusters[a], self.index.clusters[b]
        if self.p.mbr_pruning and mbr_distance(ca.bbox, cb.bbox) > self.p.delta_m:
            result = False
            self.stats['cluster_mbr_rejections'] += 1
        else:
            for c in (ca, cb):
                if c.cid not in self.trees:
                    self.trees[c.cid] = cKDTree(c.points)
            da = self.trees[b].query(ca.points, k=1)[0]
            db = self.trees[a].query(cb.points, k=1)[0]
            if self.p.metric == 'hd':
                result = max(float(da.max()), float(db.max())) <= self.p.delta_m
            else:
                # Section 5.3 interpretation: >= tau points on EACH side have
                # a neighbor within delta. Not one-to-one bipartite matching.
                result = (np.count_nonzero(da <= self.p.delta_m) >= self.p.tau and
                          np.count_nonzero(db <= self.p.delta_m) >= self.p.tau)
            self.stats['exact_cluster_comparisons'] += 1
        # Bound memoization on long histories; clearing changes speed, not results.
        if len(self.cache) >= self.p.similarity_cache_limit:
            self.cache.clear()
        if len(self.trees) >= self.p.similarity_cache_limit:
            self.trees.clear()
        self.cache[key] = bool(result)
        return bool(result)

    def convoy_bbox(self, g):
        boxes = [self.index.clusters[c].bbox for c in g.clusters]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def witness(self, a, b):
        """Return subsequence offsets proving Definition 2, or None.

        DP computes consecutive TRUE runs along diagonals. It tests ALL shifts,
        not only equal offsets, centroids, object overlap, or one matching frame.
        """
        if min(len(a.objects), len(b.objects)) < self.p.tau or min(len(a.clusters), len(b.clusters)) < self.p.k:
            return None
        if self.p.mbr_pruning and mbr_distance(self.convoy_bbox(a), self.convoy_bbox(b)) > self.p.delta_m:
            self.stats['convoy_mbr_rejections'] += 1
            return None
        previous = [0] * (len(b.clusters) + 1)
        for i, ca in enumerate(a.clusters):
            current = [0] * (len(b.clusters) + 1)
            for j, cb in enumerate(b.clusters):
                if self.cluster_similar(ca, cb):
                    current[j + 1] = previous[j] + 1
                    if current[j + 1] >= self.p.k:
                        return (i - self.p.k + 1, j - self.p.k + 1)
            previous = current
        return None


# ============================================================
# Recurrent convoy query: exact start-gap graph (Definitions 3/4)
# ============================================================
def query_recurrent(index, convoys, p, query_end):
    """Search historic predecessors lazily for anchors fully inside the window.

    Returns a DAG, not connected components: similarity is not transitive.
    Each path (oldest -> newest anchor) is a recurrent convoy sequence.
    A DAG represents ALL such paths without exponential sequence enumeration.
    Unlike literal Algorithm 3, move the frontier to EACH historical occurrence
    and check its actual start difference, including the rho boundary.
    """
    sim = Similarity(index, p)
    window_first = query_end - math.ceil(p.window_seconds / p.step_seconds) + 1
    anchors = [i for i, g in enumerate(convoys)
               if g.start >= window_first and g.end <= query_end]
    starts = [g.start for g in convoys]
    incoming, edges, visited = defaultdict(list), [], set()
    # A necessary geometric condition: enclosing circles must be within delta.
    # This safely prunes distant routes before expensive sequence comparisons.
    boxes = np.array([sim.convoy_bbox(g) for g in convoys], dtype=float).reshape(-1, 4)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    radii = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1) / 2
    spatial_tree = cKDTree(centers) if len(convoys) else None
    max_radius = float(radii.max()) if len(radii) else 0.0
    pending = list(anchors)
    rho_ticks = p.rho_seconds / p.step_seconds
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if len(visited) % 1000 == 0:
            print(f'[Recurrence] {len(visited):,} occurrences checked; {len(edges):,} links', flush=True)
        g = convoys[current]
        lo = bisect.bisect_left(starts, g.start - rho_ticks)
        hi = bisect.bisect_left(starts, g.start)  # strict chronological order
        candidates = range(lo, hi)
        if p.mbr_pruning and spatial_tree is not None:
            nearby = spatial_tree.query_ball_point(
                centers[current], radii[current] + max_radius + p.delta_m + 1e-7)
            candidates = sorted(i for i in nearby if lo <= i < hi)
            sim.stats['spatial_candidate_rejections'] += (hi - lo) - len(candidates)
        for historic in candidates:
            h = convoys[historic]
            if not p.allow_overlap and h.end >= g.start:
                continue
            sim.stats['candidate_convoy_pairs'] += 1
            witness = sim.witness(h, g)
            if witness is None:
                continue
            incoming[current].append(historic)
            edges.append({'historical_id': historic, 'later_id': current,
                          'start_gap_seconds': (g.start - h.start) * p.step_seconds,
                          'historical_match_offset': witness[0],
                          'later_match_offset': witness[1], 'match_samples': p.k})
            pending.append(historic)
    summaries = []
    # Dynamic programming over chronological IDs; no recursive depth limit.
    lengths, best_prev = {}, {}
    for node in sorted(visited):
        parents = incoming[node]
        parent = max(parents, key=lambda n: (lengths[n], -n)) if parents else None
        lengths[node] = 1 + (lengths[parent] if parent is not None else 0)
        best_prev[node] = parent
    for anchor in anchors:
        if not incoming[anchor]:
            continue  # a single occurrence is not reported as a recurrence
        chain, node = [], anchor
        while node is not None:
            chain.append(node)
            node = best_prev[node]
        summaries.append({'anchor_id': anchor, 'longest_occurrence_count': lengths[anchor],
                          'example_chain_oldest_first': list(reversed(chain)),
                          'direct_predecessor_ids': sorted(incoming[anchor])})
    return summaries, sorted(edges, key=lambda e: (e['later_id'], e['historical_id'])), sim.stats


# ============================================================
# Output, stage timing, and main program (same workflow as CuTS)
# ============================================================
def save_results(index, convoys, summaries, edges, cluster_rows, p, times, stats, projection, end):
    out = Path(p.output)
    rows = []
    for cid, g in enumerate(convoys):
        rows.append({'convoy_id': cid, 'start_time': tick_time(g.start, p),
                     'end_time': tick_time(g.end, p), 'lifetime_points': len(g.clusters),
                     'elapsed_seconds': (g.end - g.start) * p.step_seconds,
                     'inclusive_sample_seconds': len(g.clusters) * p.step_seconds,
                     'object_count': len(g.objects), 'objects': json.dumps(sorted(g.objects)),
                     'cluster_ids': json.dumps(g.clusters)})
    pd.DataFrame(rows, columns=['convoy_id','start_time','end_time','lifetime_points',
        'elapsed_seconds','inclusive_sample_seconds','object_count','objects','cluster_ids']).to_csv(out/'convoys.csv', index=False)
    pd.DataFrame(edges, columns=['historical_id','later_id','start_gap_seconds',
        'historical_match_offset','later_match_offset','match_samples']).to_csv(out/'recurrence_edges.csv', index=False)
    sr = [{**r, 'example_chain_oldest_first': json.dumps(r['example_chain_oldest_first']),
           'direct_predecessor_ids': json.dumps(r['direct_predecessor_ids'])} for r in summaries]
    pd.DataFrame(sr, columns=['anchor_id','longest_occurrence_count',
        'example_chain_oldest_first','direct_predecessor_ids']).to_csv(out/'recurrent_convoys.csv', index=False)
    pd.DataFrame(cluster_rows, columns=['time','objects','clusters','seconds']).to_csv(out/'snapshot_statistics.csv', index=False)
    # Full geometry lets the user audit every similarity witness, not just centers.
    with (out/'clusters.jsonl').open('w', encoding='utf-8') as f:
        for c in index.clusters.values():
            f.write(json.dumps({'cluster_id': c.cid, 'time': tick_time(c.tick, p),
                'objects': sorted(c.objects), 'points_m': c.points.tolist(),
                'bbox_m': c.bbox, 'index_chain_id': index.chain_id[c.cid]}) + '\n')
    report = {'parameters': asdict(p), 'projection': projection, 'index_backend': index.backend,
              'query_end': tick_time(end, p), 'timings_seconds': times,
              'counters': dict(stats), 'convoy_count': len(convoys),
              'recurrent_anchor_count': len(summaries), 'recurrence_edge_count': len(edges),
              'interpretation': 'Offline replay of observed samples. k is a sample count; rho is an inclusive start-to-start maximum. Geometry uses full DBSCAN clusters. Labels are not used. Paper candidate policy is not exhaustive subsequence enumeration.'}
    (out/'run_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def run(p):
    p.validate()
    Path(p.output).mkdir(parents=True, exist_ok=True)
    total = time.perf_counter()
    times, stats = {}, Counter()
    t = time.perf_counter()
    if p.stream_input:
        index, cluster_rows, first, end, projection = stream_clusters(p)
        times['stream_load_and_clustering'] = time.perf_counter() - t
    else:
        df, first, end, projection = load_dataset(p)
        times['load_clean_project_sample'] = time.perf_counter() - t
        t = time.perf_counter()
        index, cluster_rows = generate_clusters(df, p)
        times['snapshot_clustering'] = time.perf_counter() - t
        del df
    if p.window_seconds == 0:
        p = replace(p, window_seconds=(end - first + 1) * p.step_seconds)
    print(f"Minimum duration: {p.k} samples = {(p.k-1)*p.step_seconds} elapsed seconds")
    print(f"Recurrence start-gap limit: {p.rho_seconds / 3600:g} hours; overlap: {p.allow_overlap}")
    t = time.perf_counter()
    build_index(index, p)
    times['index_build'] = time.perf_counter() - t
    t = time.perf_counter()
    convoys = mine_convoys(index, p, first, end, stats)
    times['convoy_generation'] = time.perf_counter() - t
    t = time.perf_counter()
    summaries, edges, sim_stats = query_recurrent(index, convoys, p, end)
    stats.update(sim_stats)
    times['recurrence_query'] = time.perf_counter() - t
    t = time.perf_counter()
    save_results(index, convoys, summaries, edges, cluster_rows, p, times, stats, projection, end)
    times['export'] = time.perf_counter() - t
    times['total'] = time.perf_counter() - total
    # Refresh only timings after export, preserving the rest of the report.
    summary_path = Path(p.output)/'run_summary.json'
    report = json.loads(summary_path.read_text(encoding='utf-8'))
    report['timings_seconds'] = times
    summary_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\nConvoy occurrences: {len(convoys):,}")
    print(f"Recurrent anchors in query window: {len(summaries):,}")
    print(f"Historical links: {len(edges):,}")
    print('\nTIMING RESULTS')
    for stage, seconds in times.items():
        print(f'{stage:<30} {seconds:12.4f} s')
    print(f'\nSaved results to: {Path(p.output).resolve()}')
    return index, convoys, summaries, edges


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, value in asdict(DEFAULTS).items():
        option = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(option, type=(float if name == 'min_duration_seconds' else type(value) if value is not None else str), default=value)
    args = parser.parse_args()
    supplied = {arg.split('=')[0] for arg in sys.argv[1:]}
    if '--k' in supplied and '--min-duration-seconds' in supplied:
        parser.error('Choose --k (samples) OR --min-duration-seconds (elapsed seconds)')
    return Parameters(**vars(args))


if __name__ == '__main__':
    params = parse_args()
    Path(params.output).mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (Path(params.output)/'terminal_output.txt').open('w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
        try:
            run(params)
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
