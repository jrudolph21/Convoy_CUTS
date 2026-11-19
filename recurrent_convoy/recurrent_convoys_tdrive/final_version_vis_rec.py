#!/usr/bin/env python3
"""Interactive Folium map of recurrent convoys AND noisy Beijing taxi tracks.

Install:
    pip install pandas folium

Use with the synthetic six-month generator:
    python vis_rec_convoy.py \
        --input example_seed11/trajectories.csv \
        --convoys example_seed11/events.csv \
        --date 2008-02-02

The HTML has toggleable layers for each recurring convoy family, nearby
random distractor tracks (orange), and background/random taxis (gray).
If a manifest.json lives next to trajectories.csv, the generator's distractor
IDs are automatically identified; otherwise all non-convoy points are gray.

For one recurrent group from your DETECTOR (results folder):
    python vis_rec_convoy.py --results synthetic_results_4hr --list-anchors
    python vis_rec_convoy.py --results synthetic_results_4hr \
        --input seed11_every_second/trajectories.csv \
        --anchor-id 16 --output anchor_16_full_day.html
The selected anchor's own date is used if --date is omitted. Show each
anchor member's full-day GPS track; highlight all same-day linked occurrences.
The results folder does NOT contain the original GPS input; provide it via --input.

To map ALL detected recurrence-linked occurrences involving selected taxis:
    python vis_rec_convoy.py --results synthetic_results_4hr \
        --input seed11_every_second/trajectories.csv \
        --taxi-ids 101 102 103 104 --match all --date 2008-01-01 \
        --output four_taxis.html
--match all requires all IDs on both sides of a recurrence link;
--match any requires at least one specified taxi on both sides.
This filters actual detector recurrence_edges.csv, not planted event labels.

To compare two recurrence-linked detector occurrences (NO raw GPS input needed):
    python vis_all_rec_taxi.py --results tdrive_results_4_4_24 \
        --historical-id 14 --later-id 104 \
        --output recurrence_14_104.html
This draws one toggleable layer per convoy, in two distinct colors.
The precise linked cluster snapshots are emphasized with thicker lines.
Use --input TRAJECTORIES.csv to overlay every member taxi's complete daily GPS
route, with a separate toggle for each taxi. --pair-focus day (automatic with
--input) shows the entire day's movement; match zooms to recurrence geometry;
all zooms to the full detected occurrences.
This mode uses detector clusters.jsonl and run_summary.json directly.
Taxi membership can differ entirely between the two convoys.

Synthetic events.csv is PLANTED ground truth, not discovered detector output.
With events.csv, a family is recurrent when its family_id appears at least
TWICE in the full events.csv (not necessarily twice on the chosen day).
With a detector convoys.csv, use --edges recurrence_edges.csv to show only
linked occurrences. Without family_id or --edges, all listed occurrences
are plotted and recurrence is *not verified*.

Noise is defined as observations OUTSIDE labeled convoy intervals, including
independent distractor trajectories. This avoids erroneously treating a taxi
as permanently part of a convoy simply because it participates once.

Default map window is one day. Noise trajectories are sampled and capped for
browser performance; run --help for controls. x = longitude, y = latitude.
All input times are assumed to be timezone-naive Beijing local times.



python final_version_vis_rec.py --results tdrive_new_results --input tdrive_interpolated.csv --historical-id 12 --later-id 35 --pair-focus day --output new_recurrence_map.html

"""

import argparse
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import random

import folium
import pandas as pd


CARTO_API_KEY = "cb1_3wq7_1_44372b8a982bea61dc3eedfd"

carto_tiles = (
    "https://basemaps.cartocdn.com/rastertiles/"
    "light_all/{z}/{x}/{y}.png"
    f"?key={CARTO_API_KEY}"
)

# Colors are assigned consistently to families across daily maps.
COLORS = [
    '#de2d26', '#2064f5', '#16a34a', '#9333ea', '#e11d86', '#e58b00',
    '#0d9488', '#92400e', '#4f46e5', '#a21caf', '#d44812', '#0077aa',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', help='Original id,time,x,y trajectory CSV (optional for pair mode; adds complete daily routes)')
    p.add_argument('--convoys', help='Planted events.csv or detector convoys.csv (existing multi-convoy mode)')
    p.add_argument('--results', help='Detector output directory containing convoys.csv and recurrent_convoys.csv')
    p.add_argument('--anchor-id', help='One recurrent anchor ID from recurrent_convoys.csv')
    p.add_argument('--historical-id', help='Earlier convoy occurrence ID from recurrence_edges.csv')
    p.add_argument('--later-id', '--latter-id', dest='later_id',
                   help='Later convoy occurrence ID from recurrence_edges.csv')
    p.add_argument('--pair-focus', choices=('match', 'all', 'day'), default=None,
                   help='Initial extent for a pair: matched snapshots, full detected occurrences, or full-day taxi routes; default day with --input, otherwise match')
    p.add_argument('--taxi-ids', nargs='+', metavar='ID',
                   help='Highlight all recurrence-linked occurrences involving these taxi IDs')
    p.add_argument('--match', choices=('all', 'any'), default='all',
                   help='For --taxi-ids: require every ID, or at least one shared ID (default: all)')
    p.add_argument('--list-anchors', action='store_true', help='List detected recurrent anchors and member IDs')
    p.add_argument('--track-gap-seconds', type=float, default=360,
                   help='Break full-day taxi tracks after missing observations longer than this (default 360)')
    p.add_argument('--gps-dots', type=int, default=80,
                   help='Maximum observation dots per taxi route; polylines retain all points (default 80)')
    p.add_argument('--date', help='Single Beijing-local day, e.g. 2008-02-02')
    p.add_argument('--start', help='Optional window start, e.g. "2008-02-02 09:00:00"')
    p.add_argument('--end', help='Optional inclusive window end')
    p.add_argument('--edges', help='Optional recurrence edges (detector or synthetic format)')
    p.add_argument('--target-index', type=int, help='Only plot this CSV row (zero-based), plus noise')
    p.add_argument('--output', default='convoy_map.html', help='Output HTML map path')
    p.add_argument('--manifest', help='Generator manifest.json (auto-detected beside --input)')
    p.add_argument('--gap-seconds', type=float, default=2,
                   help='Break convoy segments at larger gaps (default 2)')
    p.add_argument('--noise-gap-seconds', type=float, default=None,
                   help='Break noise tracks at larger gaps (auto from generator manifest)')
    p.add_argument('--noise-sample-seconds', type=float, default=0,
                   help='Minimum time between displayed noise points, retaining endpoints')
    p.add_argument('--max-noise-taxis', type=int, default=300,
                   help='Limit noise taxis shown (prefer nearby distractors; default 300)')
    p.add_argument('--max-noise-points', type=int, default=35000,
                   help='Maximum rendered noise observations (default 35000)')
    p.add_argument('--seed', type=int, default=11, help='Reproducible noise subset')
    p.add_argument('--noise-window-seconds', type=float, default=0,
                   help='If >0, only noise within +/- this many seconds of a convoy event')
    p.add_argument('--focus', choices=['all', 'convoys', 'noise', 'taxi'], default='all',
                   help='Initial map extent (default: all plotted tracks)')
    p.add_argument('--taxi-id', type=str, help='Highlight this taxi entire daily route with start/end and GPS dots')
    p.add_argument('--no-noise', action='store_true',
                   help='Hide noise entirely (optional)')
    p.add_argument('--carto-api-key', help='Optional CARTO raster tile key')
    return p.parse_args()


def parse_time(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is not None:
        raise ValueError(f'Invalid or timezone-aware timestamp: {value!r}')
    return t


def read_objects(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'Expected JSON list of taxi IDs, got {value!r}')
    return [str(v) for v in value]


def load_events(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {'start_time', 'end_time'} <= set(table.columns):
        raise ValueError('Convoys CSV needs start_time,end_time')
    if 'object_ids' not in table.columns and 'objects' not in table.columns:
        raise ValueError('Convoys CSV needs object_ids or objects')
    events = []
    for idx, row in enumerate(table.to_dict('records')):
        begin, end = parse_time(row['start_time']), parse_time(row['end_time'])
        if end < begin:
            raise ValueError(f'Event at CSV row {idx} ends before it starts')
        ids = read_objects(row.get('object_ids') or row.get('objects'))
        event_id = str(row.get('event_id') or row.get('convoy_id') or idx)
        family_id = str(row.get('family_id') or '')
        events.append({
            **row,
            '_index': idx,
            '_event_id': event_id,
            '_family_id': family_id,
            '_ids': ids,
            '_start': begin,
            '_end': end,
        })
    return events, ('object_ids' in table.columns)


def recurring_identifiers(events, edges_path):
    """Return recurrence-linked IDs; None means recurrence is unknown."""
    if edges_path:
        edges = pd.read_csv(edges_path, dtype=str, keep_default_na=False)
        candidates = (
            ('historical_event_id', 'later_event_id'),  # generator expected_links.csv
            ('historical_id', 'later_id'),              # detector recurrence_edges.csv
            ('source_id', 'target_id'),
        )
        for left, right in candidates:
            if {left, right} <= set(edges.columns):
                return set(edges[left]) | set(edges[right]), 'edges'
        raise ValueError('Unrecognized --edges columns; expected historical_event_id/later_event_id '
                         'or historical_id/later_id')

    counts = Counter(e['_family_id'] for e in events if e['_family_id'])
    if counts:
        return {e['_event_id'] for e in events if e['_family_id'] and
                counts[e['_family_id']] >= 2}, 'family_id'
    return None, 'unverified'


def window_bounds(args, events):
    if args.date:
        day = parse_time(args.date).normalize()
    elif args.start:
        day = parse_time(args.start).normalize()
    elif args.end:
        day = parse_time(args.end).normalize()
    elif events:
        day = min(e['_start'] for e in events).normalize()
    else:
        raise ValueError('Supply --date or --start when the convoys CSV is empty')

    start, end = day, day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if args.start:
        start = max(start, parse_time(args.start)) if args.date else parse_time(args.start)
    if args.end:
        end = min(end, parse_time(args.end)) if args.date else parse_time(args.end)
    if end < start:
        raise ValueError('The selected --date, --start, and --end do not overlap')
    return start, end


def load_manifest(args):
    path = Path(args.manifest) if args.manifest else Path(args.input).with_name('manifest.json')
    if not path.is_file():
        if args.manifest:
            raise FileNotFoundError(f'Manifest not found: {path}')
        return set(), False, 12.0
    with path.open('r', encoding='utf-8') as f:
        manifest = json.load(f)
    ids = set()
    for family in manifest.get('family_details', []):
        ids.update(str(item) for item in family.get('distractor_ids', []))
    print(f'Read {len(ids)} distractor taxi IDs from {path}')
    params = manifest.get('parameters', {})
    cadence = params.get('track_cadence', params.get('background_cadence', 5))
    try:
        gap = max(2.0, float(cadence) * 1.7)
    except (TypeError, ValueError):
        gap = 12.0
    return ids, True, gap


def load_window(input_path, start, end):
    """Stream CSV chunks; keep only the requested times, not six months in RAM."""
    chunks = []
    scanned = 0
    for chunk in pd.read_csv(input_path, usecols=['id', 'time', 'x', 'y'],
                             dtype={'id': str, 'time': str}, chunksize=200_000):
        scanned += len(chunk)
        # The generator uses ISO-like sortable time strings. The cheap string
        # prefilter avoids parsing ~2M timestamps when only one day is wanted.
        lo = start.strftime('%Y-%m-%d %H:%M:%S')[:10]
        hi = end.strftime('%Y-%m-%d %H:%M:%S')[:10]
        rough = chunk[chunk['time'].str[:10].between(lo, hi)].copy()
        if rough.empty:
            continue
        rough['time'] = pd.to_datetime(rough['time'], errors='coerce')
        rough['x'] = pd.to_numeric(rough['x'], errors='coerce')
        rough['y'] = pd.to_numeric(rough['y'], errors='coerce')
        rough = rough.dropna(subset=['id', 'time', 'x', 'y'])
        rough = rough[rough['time'].between(start, end) &
                      rough['x'].between(-180, 180) & rough['y'].between(-90, 90)]
        if not rough.empty:
            chunks.append(rough)
    print(f'Scanned {scanned:,} CSV observations; kept {sum(len(c) for c in chunks):,} in selected window')
    if not chunks:
        return pd.DataFrame(columns=['id', 'time', 'x', 'y'])
    return pd.concat(chunks, ignore_index=True).sort_values(['id', 'time']).reset_index(drop=True)


def event_rows(data, event):
    return data[data['id'].isin(event['_ids']) &
                data['time'].between(event['_start'], event['_end'])]


def classify_observations(data, all_events):
    """Exclude ONLY the samples within ALL labeled events from noise."""
    marked = pd.Series(False, index=data.index)
    for event in all_events:
        marked |= (data['id'].isin(event['_ids']) &
                   data['time'].between(event['_start'], event['_end']))
    return data.loc[~marked].copy()


def restrict_noise_to_events(noise, events, window_seconds):
    if window_seconds == 0 or not events or noise.empty:
        return noise
    nearby = pd.Series(False, index=noise.index)
    padding = pd.Timedelta(seconds=window_seconds)
    for e in events:
        nearby |= noise['time'].between(e['_start'] - padding, e['_end'] + padding)
    return noise.loc[nearby]


def select_noise_taxis(noise, distractor_ids, max_taxis, seed):
    rng = random.Random(seed)
    ids = set(noise['id'].unique())
    clutter = sorted(ids & distractor_ids)
    background = sorted(ids - distractor_ids)
    if len(clutter) >= max_taxis:
        keep = set(rng.sample(clutter, max_taxis))
    else:
        keep = set(clutter)
        quota = max_taxis - len(keep)
        keep.update(rng.sample(background, min(quota, len(background))))
    return noise[noise['id'].isin(keep)].copy()


def thin_segment(segment, sample_seconds):
    if sample_seconds == 0 or len(segment) <= 2:
        return segment
    stamps = segment['time'].tolist()
    keep = [0]
    last = stamps[0]
    for i in range(1, len(stamps) - 1):
        if (stamps[i] - last).total_seconds() >= sample_seconds:
            keep.append(i)
            last = stamps[i]
    keep.append(len(segment) - 1)
    return segment.iloc[keep]


def noise_segments(df, gap_seconds, sample_seconds):
    """Create independent segments: don't connect across observation gaps."""
    for oid, taxi in df.groupby('id', sort=True):
        breakpoints = taxi['time'].diff().dt.total_seconds().gt(gap_seconds).cumsum()
        for _, seg in taxi.groupby(breakpoints, sort=False):
            yield oid, thin_segment(seg, sample_seconds)


def clip_segment_to_budget(seg, remaining):
    if len(seg) <= remaining:
        return seg
    if remaining < 2:
        return seg.iloc[0:0]
    # Evenly spread a limited number of points, always including endpoints.
    indices = sorted({round(i * (len(seg) - 1) / (remaining - 1))
                      for i in range(remaining)})
    return seg.iloc[indices]


def make_map(args):
    for name in ('gap_seconds',):
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if args.noise_sample_seconds < 0 or args.noise_window_seconds < 0:
        raise ValueError('Noise sampling and noise window must be nonnegative')
    if args.max_noise_taxis < 1 or args.max_noise_points < 1:
        raise ValueError('Noise caps must be positive')

    events, is_planted = load_events(args.convoys)
    if args.target_index is not None and not 0 <= args.target_index < len(events):
        raise ValueError(f'--target-index must be in [0, {len(events) - 1}]')
    start, end = window_bounds(args, events)
    day_events = [e for e in events if e['_start'] <= end and e['_end'] >= start]
    allowed, recurrence_method = recurring_identifiers(events, args.edges)
    chosen = [e for e in day_events if allowed is None or e['_event_id'] in allowed]
    if args.target_index is not None:
        chosen = [e for e in chosen if e['_index'] == args.target_index]

    data = load_window(args.input, start, end)
    if data.empty:
        raise ValueError('No GPS observations for the selected date/window')

    # Sample NON-convoy traffic across the day, including independent near-event
    # distractors. All planted event observations are excluded, even if just one
    # chosen family is selected for visualization.
    noise = classify_observations(data, day_events)
    raw_noise = len(noise)
    noise = restrict_noise_to_events(noise, chosen, args.noise_window_seconds)
    distractor_ids, has_manifest, suggested_gap = load_manifest(args)
    if args.noise_gap_seconds is None:
        args.noise_gap_seconds = suggested_gap
    if args.noise_gap_seconds <= 0:
        raise ValueError('--noise-gap-seconds must be positive')
    if args.taxi_id is not None:
        noise = noise[noise['id'] != str(args.taxi_id)].copy()
    if not args.no_noise and not noise.empty:
        noise = select_noise_taxis(noise, distractor_ids, args.max_noise_taxis, args.seed)
    else:
        noise = noise.iloc[0:0]

    tiles = 'CartoDB positron'
    m = folium.Map(location=[39.90, 116.40], zoom_start=12,
                    tiles=carto_tiles, attr="© OpenStreetMap contributors, © CARTO", control_scale=True, prefer_canvas=True)
    # folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    # if args.carto_api_key:
    #     from urllib.parse import quote
    #     folium.TileLayer(
    #         tiles=('https://basemaps.cartocdn.com/rastertiles/light_all/'
    #                '{z}/{x}/{y}.png?key=' + quote(args.carto_api_key, safe='')),
    #         attr='&copy; OpenStreetMap contributors &copy; CARTO',
    #         name='CARTO Light (key)', max_zoom=20).add_to(m)

    all_bounds, convoy_bounds, noise_bounds = [], [], []
    noise_stats = {'tracks': set(), 'segments': 0, 'points': 0, 'clutter': 0, 'background': 0}

    # Draw noise first, beneath highlighted convoy routes.
    bg_layer = folium.FeatureGroup(name='Noise · background taxis (gray)', show=True)
    clutter_layer = folium.FeatureGroup(name='Noise · nearby distractors (orange)', show=True)
    # Allocate the point budget to nearby distractors FIRST, so a large
    # background population does not hide the noise right around convoys.
    near_noise = noise[noise['id'].isin(distractor_ids)]
    other_noise = noise[~noise['id'].isin(distractor_ids)]
    for category, noise_part in (('clutter', near_noise), ('background', other_noise)):
        for oid, seg in noise_segments(noise_part, args.noise_gap_seconds,
                                       args.noise_sample_seconds):
            remaining = args.max_noise_points - noise_stats['points']
            if remaining <= 0:
                break
            seg = clip_segment_to_budget(seg, remaining)
            if seg.empty:
                break
            target = clutter_layer if category == 'clutter' else bg_layer
            color = '#eb9b42' if category == 'clutter' else '#6e7784'
            opacity = 0.65 if category == 'clutter' else 0.27
            coords = [(float(row.y), float(row.x)) for row in seg.itertuples()]
            noise_bounds.extend(coords)
            label = ('Nearby independent taxi' if category == 'clutter'
                     else 'Random background taxi')
            tooltip = html.escape(f'{label} {oid} | '
                                  f'{seg.time.iloc[0]} to {seg.time.iloc[-1]}')
            if len(coords) > 1:
                folium.PolyLine(coords, color=color, weight=2, opacity=opacity,
                                tooltip=tooltip).add_to(target)
            else:
                folium.CircleMarker(coords[0], radius=2, color=color,
                                    fill=True, fill_opacity=opacity, opacity=opacity,
                                    tooltip=tooltip).add_to(target)
            noise_stats['segments'] += 1
            noise_stats['points'] += len(seg)
            noise_stats['tracks'].add(oid)
            noise_stats[category] += len(seg)

    bg_layer.add_to(m)
    if has_manifest and distractor_ids:
        clutter_layer.add_to(m)

    taxi_bounds = []
    if args.taxi_id is not None:
        whole_taxi = data[data['id'] == args.taxi_id].sort_values('time')
        if whole_taxi.empty:
            raise ValueError(f'No observations for taxi {args.taxi_id} on this date')
        track_layer = folium.FeatureGroup(
            name=html.escape(f'Daily route: taxi {args.taxi_id} (blue with GPS dots)'),
            show=True)
        for oid, seg in noise_segments(whole_taxi, args.noise_gap_seconds, 0):
            coords = [(float(p.y), float(p.x)) for p in seg.itertuples()]
            taxi_bounds.extend(coords)
            if len(coords) > 1:
                folium.PolyLine(coords, color='#3f52db', weight=4, opacity=0.75,
                                tooltip=f'Taxi {html.escape(oid)} full-day route').add_to(track_layer)
        # Use dots at regular intervals; leave every point in the polyline.
        dot_step = max(1, len(whole_taxi) // 110)
        for p in whole_taxi.iloc[::dot_step].itertuples():
            folium.CircleMarker((float(p.y), float(p.x)), radius=3,
                                color='#111111', weight=1.5,
                                fill=True, fill_color='#f22b2b', fill_opacity=0.95,
                                tooltip=f'Taxi {args.taxi_id} · {p.time}').add_to(track_layer)
        firstp, lastp = whole_taxi.iloc[0], whole_taxi.iloc[-1]
        folium.Marker((float(firstp.y), float(firstp.x)),
                      icon=folium.Icon(color='green', icon='play'),
                      tooltip=f'Taxi {args.taxi_id} START: {firstp.time}').add_to(track_layer)
        folium.Marker((float(lastp.y), float(lastp.x)),
                      icon=folium.Icon(color='red', icon='stop'),
                      tooltip=f'Taxi {args.taxi_id} END: {lastp.time}').add_to(track_layer)
        track_layer.add_to(m)

    family_events = defaultdict(list)
    for e in chosen:
        key = e['_family_id'] if e['_family_id'] else e['_event_id']
        family_events[key].append(e)

    actual_convoys = 0
    actual_convoy_points = 0
    for group_key, entries in family_events.items():
        e0 = entries[0]
        family_number = (int(e0['_family_id']) if e0['_family_id'].isdigit()
                         else e0['_index'])
        color = COLORS[family_number % len(COLORS)]
        period_sec = e0.get('interval_seconds', '')
        if period_sec.isdigit():
            period = (f'every {int(period_sec) // 86400} day(s)' if int(period_sec) >= 86400
                      else f'every {int(period_sec) / 3600:g}h')
        else:
            period = 'interval unknown'
        group_name = (f'Convoy family {group_key} · {len(entries)} events · {period}'
                      if e0['_family_id'] else f'Convoy {group_key}')
        layer = folium.FeatureGroup(name=html.escape(group_name), show=True)
        for e in entries:
            part = event_rows(data, e)
            if part.empty:
                continue
            actual_convoys += 1
            actual_convoy_points += len(part)
            for oid, taxi in part.groupby('id', sort=True):
                taxi = taxi.sort_values('time')
                splits = taxi.time.diff().dt.total_seconds().gt(args.gap_seconds).cumsum()
                for _, segment in taxi.groupby(splits, sort=False):
                    coords = list(zip(segment['y'].astype(float), segment['x'].astype(float)))
                    convoy_bounds.extend(coords)
                    tt = html.escape(f'Family {group_key} / Event {e["_event_id"]} / '
                                     f'Taxi {oid} / {e["_start"]} – {e["_end"]}')
                    if len(coords) > 1:
                        folium.PolyLine(coords, color=color, weight=5, opacity=0.9,
                                        tooltip=tt).add_to(layer)
                    else:
                        folium.CircleMarker(coords[0], radius=3, color=color,
                                            fill=True, fill_opacity=1,
                                            tooltip=tt).add_to(layer)

            # A visible clickable marker is needed because a 2-8 second path
            # is only a few dozen meters long at Beijing-wide zoom levels.
            first_rows = part[part['time'] == part['time'].min()]
            marker_loc = [float(first_rows['y'].mean()), float(first_rows['x'].mean())]
            content = (
                f'<b>Family {html.escape(str(group_key))} / Event '
                f'{html.escape(e["_event_id"])}</b><br>'
                f'{html.escape(str(e["_start"]))} to {html.escape(str(e["_end"]))}<br>'
                f'Convoy members: {len(e["_ids"])}<br>'
                f'IDs: {html.escape(", ".join(e["_ids"]))}<br>'
                f'Recurrence: {html.escape(period)}'
            )
            marker = folium.CircleMarker(
                marker_loc, radius=7, color=color,
                fill=True, fill_color=color, fill_opacity=0.85,
                weight=2, popup=folium.Popup(content, max_width=360),
                tooltip=html.escape(f'Convoy {e["_event_id"]}: '
                                    f'{e["_start"].strftime("%H:%M:%S")}'))
            marker.add_to(layer)
            convoy_bounds.append(marker_loc)
        layer.add_to(m)

    if args.focus == 'taxi' and taxi_bounds:
        all_bounds = taxi_bounds
    elif args.focus == 'convoys' and convoy_bounds:
        all_bounds = convoy_bounds
    elif args.focus == 'noise' and noise_bounds:
        all_bounds = noise_bounds
    else:
        all_bounds = noise_bounds + convoy_bounds
    if all_bounds:
        mins = [min(pt[0] for pt in all_bounds), min(pt[1] for pt in all_bounds)]
        maxs = [max(pt[0] for pt in all_bounds), max(pt[1] for pt in all_bounds)]
        if mins == maxs:
            m.location = mins
            m.zoom_start = 15
        else:
            m.fit_bounds([mins, maxs], max_zoom=17, padding=(25, 25))

    nature = 'Planted synthetic events' if is_planted else 'Detector convoy occurrences'
    status = ('recurrence from ' + recurrence_method if recurrence_method != 'unverified'
              else 'recurrence not verified (no edges / family_id)')
    summary = (
        '<div style="position:fixed;top:12px;left:55px;z-index:9999;'
        'background:rgba(255,255,255,.96);border:1px solid #aaa;'
        'padding:10px 13px;border-radius:6px;box-shadow:0 2px 7px #0002;'
        'font:12px Arial;max-width:340px;pointer-events:none">'
        '<b style="font-size:14px">Beijing: recurrent convoys + noise</b><br>'
        f'{html.escape(str(start))} to {html.escape(str(end))}<br>'
        f'{html.escape(nature)} · {html.escape(status)}<br>'
        f'<span style="color:#c32621"><b>{actual_convoys}</b> highlighted convoy occurrences</span><br>'
        f'<span style="color:#6e7784">{len(noise_stats["tracks"])} noise taxis; '
        f'{noise_stats["segments"]} displayed track segments</span><br>'
        f'<span style="color:#e58b00">Orange = independent nearby distractors</span><br>'
        'Gray = other random taxi movements<br>'
        '<small>Use layer control to show / hide families and noise.</small>'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(summary))
    folium.LayerControl(collapsed=False).add_to(m)

    output = Path(args.output)
    if output.suffix.lower() != '.html':
        output = Path(str(output) + '.html')
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    print(f'Selected day: {len(day_events)} labeled events; '
          f'{len(chosen)} recurrent/selected events; {actual_convoys} with GPS rows')
    print(f'Noise: {raw_noise:,} raw non-convoy observations; '
          f'{len(noise_stats["tracks"])} displayed taxis; '
          f'{noise_stats["points"]:,} displayed points, '
          f'{noise_stats["segments"]:,} segments')
    print(f'Saved interactive map: {output.resolve()}')
    return m




# -----------------------------------------------------------------------------
# Single *detected recurrent* convoy mode. Read the user's algorithm results
# from a results directory, without ever using planted generator labels.
# -----------------------------------------------------------------------------

def read_result_table(result_path, basename):
    """Read a detector output CSV from a results directory (not a ZIP)."""
    root = Path(result_path)
    if not root.is_dir():
        raise NotADirectoryError(f'--results must be a folder: {root}')
    matches = sorted(root.rglob(basename))
    if len(matches) != 1:
        raise ValueError(f'Expected one {basename} under {root}; found {len(matches)}')
    return pd.read_csv(matches[0], dtype=str, keep_default_na=False)


def detector_tables(results_path):
    occurrences = read_result_table(results_path, 'convoys.csv')
    recurrent = read_result_table(results_path, 'recurrent_convoys.csv')
    if not {'convoy_id', 'objects', 'start_time', 'end_time'} <= set(occurrences.columns):
        raise ValueError('convoys.csv must contain convoy_id,objects,start_time,end_time')
    if not {'anchor_id', 'example_chain_oldest_first'} <= set(recurrent.columns):
        raise ValueError('recurrent_convoys.csv must contain anchor_id,example_chain_oldest_first')
    if occurrences.convoy_id.duplicated().any() or recurrent.anchor_id.duplicated().any():
        raise ValueError('Duplicate occurrence IDs or anchor IDs in detection results')
    return occurrences, recurrent


def print_recurrent_anchors(occurrences, recurrent):
    by_id = occurrences.set_index('convoy_id', drop=False)
    print(f'Found {len(recurrent)} recurrent anchors. Showing the highest-repeat ones first:')
    print('Anchor | Occurrences | Anchor time (Beijing) | Taxi IDs')
    print('-' * 90)
    rows = sorted(recurrent.to_dict('records'),
                  key=lambda r: (-int(r.get('longest_occurrence_count') or 0),
                                 int(r['anchor_id']) if r['anchor_id'].isdigit() else r['anchor_id']))
    for r in rows[:30]:
        oid = str(r['anchor_id'])
        if oid not in by_id.index:
            print(f'{oid:>6} | missing occurrence')
            continue
        selected = by_id.loc[oid]
        ids = ','.join(read_objects(selected['objects']))
        print(f'{oid:>6} | {r["longest_occurrence_count"]:>11} | '
              f'{selected["start_time"]:<22} | {ids}')
    if len(rows) > 30:
        print(f'... and {len(rows) - 30} more anchors. Use --anchor-id to select one.')


def load_daily_member_tracks(csv_path, day_start, day_end, wanted_ids):
    """Stream a possibly six-month CSV without loading all taxis into RAM."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Trajectories CSV not found: {path}. The detection folder has only results; '
            'use the ORIGINAL GPS input CSV named in run_summary.json.')
    batches = []
    scanned, kept = 0, 0
    date_str = day_start.strftime('%Y-%m-%d')
    for chunk in pd.read_csv(path, usecols=['id', 'time', 'x', 'y'],
                             dtype={'id': str, 'time': str}, chunksize=250_000):
        scanned += len(chunk)
        chunk = chunk.loc[chunk['id'].isin(wanted_ids) &
                          chunk['time'].str.startswith(date_str)].copy()
        if chunk.empty:
            continue
        chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
        chunk['x'] = pd.to_numeric(chunk['x'], errors='coerce')
        chunk['y'] = pd.to_numeric(chunk['y'], errors='coerce')
        chunk = chunk.dropna(subset=['id', 'time', 'x', 'y'])
        chunk = chunk[chunk['time'].between(day_start, day_end) &
                      chunk['x'].between(-180, 180) & chunk['y'].between(-90, 90)]
        if not chunk.empty:
            batches.append(chunk)
            kept += len(chunk)
    print(f'Scanned {scanned:,} GPS observations; loaded {kept:,} for '
          f'{len(wanted_ids)} selected taxis on {date_str}.')
    if not batches:
        return pd.DataFrame(columns=['id', 'time', 'x', 'y'])
    return (pd.concat(batches, ignore_index=True)
            .sort_values(['id', 'time']).reset_index(drop=True))


def member_track_segments(taxi, max_gap_seconds):
    """Preserve observed paths, but never connect points across long gaps."""
    taxi = taxi.sort_values('time')
    cuts = taxi['time'].diff().dt.total_seconds().gt(max_gap_seconds).cumsum()
    for _, segment in taxi.groupby(cuts, sort=False):
        yield segment


def put_polyline(group, frame, color, weight, opacity, tooltip,
                 gap_seconds, dash_array=None):
    bounds = []
    for segment in member_track_segments(frame, gap_seconds):
        coords = [(float(r.y), float(r.x)) for r in segment.itertuples()]
        bounds.extend(coords)
        if len(coords) >= 2:
            folium.PolyLine(coords, color=color, weight=weight, opacity=opacity,
                            tooltip=tooltip, dash_array=dash_array).add_to(group)
        elif coords:
            folium.CircleMarker(coords[0], radius=max(3, weight), color=color,
                                fill=True, fill_opacity=opacity,
                                tooltip=tooltip).add_to(group)
    return bounds


def one_recurrent_convoy_map(args, occurrences, recurrent):
    """Plot full-day routes of anchor's member taxis and matched recurrence events."""
    if args.input is None:
        raise ValueError('--input PATH/trajectories.csv is required to map taxi routes.')
    if args.track_gap_seconds <= 0 or args.gap_seconds <= 0:
        raise ValueError('Track and convoy gap seconds must be positive')
    if args.gps_dots < 0:
        raise ValueError('--gps-dots must be nonnegative')
    anchor_id = str(args.anchor_id)
    matching = recurrent.loc[recurrent['anchor_id'] == anchor_id]
    if matching.empty:
        raise ValueError(f'Anchor {anchor_id} not present in recurrent_convoys.csv. '
                         'Use --results ... --list-anchors to find available IDs.')
    anchor_info = matching.iloc[0]
    chain = [str(oid) for oid in read_objects(anchor_info['example_chain_oldest_first'])]
    if not chain or chain[-1] != anchor_id:
        raise ValueError(f'Unexpected recurrence chain for anchor {anchor_id}: {chain}')
    by_id = occurrences.set_index('convoy_id', drop=False)
    missing = [x for x in chain if x not in by_id.index]
    if missing:
        raise ValueError(f'Missing convoy IDs referenced by anchor {anchor_id}: {missing}')
    anchor = by_id.loc[anchor_id]
    member_ids = read_objects(anchor['objects'])
    selected_date = parse_time(args.date).normalize() if args.date else parse_time(anchor['start_time']).normalize()
    day_end = selected_date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    day_chain = [by_id.loc[oid] for oid in chain
                 if parse_time(by_id.loc[oid]['start_time']) <= day_end and
                    parse_time(by_id.loc[oid]['end_time']) >= selected_date]
    if not day_chain:
        raise ValueError(f'No occurrences in recurrence chain {chain} on {selected_date.date()}. '
                         'Try omitting --date to use the anchor\'s own day.')
    data = load_daily_member_tracks(args.input, selected_date, day_end, set(member_ids))
    if data.empty:
        raise ValueError('No matching taxi GPS observations for this day. '
                         'Check that --input is the CSV used to produce these detector results.')

    color_options = ['#2563eb', '#16a34a', '#9333ea', '#db2777', '#0d9488',
                     '#ca8a04', '#7c3aed', '#0284c7', '#a16207']
    tiles = 'CartoDB positron'
    m = folium.Map(location=[39.9, 116.4], zoom_start=11,
                    tiles=carto_tiles, attr="© OpenStreetMap contributors, © CARTO", control_scale=True, prefer_canvas=True)
    # folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)

    all_bounds = []
    for j, oid in enumerate(member_ids):
        taxi = data[data.id == oid]
        if taxi.empty:
            print(f'Warning: no GPS samples for selected member {oid} on this day.')
            continue
        color = color_options[j % len(color_options)]
        layer = folium.FeatureGroup(name=f'Taxi {html.escape(oid)} · complete daily route',
                                    show=True)
        bounds = put_polyline(layer, taxi, color, 3.5, .80,
                              f'Taxi {html.escape(oid)} · full-day GPS route',
                              args.track_gap_seconds)
        all_bounds.extend(bounds)
        if args.gps_dots:
            # Dots are a visual sample only; the route line keeps all GPS points.
            dots = min(args.gps_dots, len(taxi))
            indices = (sorted({round(k * (len(taxi)-1)/(dots-1)) for k in range(dots)})
                       if dots >= 2 else [0])
            for sample in taxi.iloc[indices].itertuples():
                folium.CircleMarker((float(sample.y), float(sample.x)),
                                    radius=2.8, color='#202020', weight=.7,
                                    fill=True, fill_color='#f33c3c', fill_opacity=.88,
                                    tooltip=f'Taxi {oid} · {sample.time}').add_to(layer)
        first, last = taxi.iloc[0], taxi.iloc[-1]
        folium.CircleMarker((float(first.y), float(first.x)), radius=7,
                            color='#15803d', fill=True, fill_color='#22c55e',
                            fill_opacity=1, weight=2,
                            tooltip=f'Taxi {oid} first observation: {first.time}').add_to(layer)
        folium.CircleMarker((float(last.y), float(last.x)), radius=7,
                            color='#b91c1c', fill=True, fill_color='#ef4444',
                            fill_opacity=1, weight=2,
                            tooltip=f'Taxi {oid} last observation: {last.time}').add_to(layer)
        layer.add_to(m)

    old = folium.FeatureGroup(name='Earlier occurrences in recurrence chain (orange)', show=True)
    current = folium.FeatureGroup(name=f'Anchor {anchor_id} detected convoy (bold red)', show=True)
    highlighted = 0
    for occ in day_chain:
        oid = str(occ['convoy_id'])
        start, end = parse_time(occ['start_time']), parse_time(occ['end_time'])
        # Some detection chains have changing membership. Show only anchor taxis'
        # GPS paths; include full declared membership in the occurrence popup.
        occurrence_ids = read_objects(occ['objects'])
        visible = set(member_ids) & set(occurrence_ids)
        part = data.loc[data.id.isin(visible) & data.time.between(start, end)]
        if part.empty:
            print(f'Warning: no member GPS observations for occurrence {oid} at {start}')
            continue
        primary = oid == anchor_id
        group = current if primary else old
        color = '#d81b34' if primary else '#f59e0b'
        for member, taxi in part.groupby('id'):
            all_bounds.extend(put_polyline(group, taxi, color,
                                           8 if primary else 6, .97,
                                           f'Detected convoy {oid} · taxi {member} · {start}–{end}',
                                           args.gap_seconds,
                                           None if primary else '6,4'))
        first = part[part.time == part.time.min()]
        centroid = [float(first.y.mean()), float(first.x.mean())]
        tooltip = f'{"Selected" if primary else "Earlier"} occurrence {oid} · {start}'
        popup = (f'<b>Detected convoy {html.escape(oid)}</b><br>'
                 f'{html.escape(str(start))} to {html.escape(str(end))}<br>'
                 f'Members: {html.escape(", ".join(occurrence_ids))}<br>'
                 f'Part of recurrence chain: {html.escape(" → ".join(chain))}')
        folium.CircleMarker(centroid, radius=9 if primary else 6,
                            color='#7f1d1d' if primary else '#b45309',
                            fill=True, fill_color=color, fill_opacity=.95,
                            tooltip=tooltip, popup=folium.Popup(popup, max_width=350)).add_to(group)
        highlighted += 1
    if highlighted == 0:
        raise ValueError('No selected convoy GPS points match the detector timestamps. '
                         'Check that the original CSV matches the detector results.')
    old.add_to(m)
    current.add_to(m)

    if all_bounds:
        m.fit_bounds([[min(lat for lat, _ in all_bounds), min(lon for _, lon in all_bounds)],
                      [max(lat for lat, _ in all_bounds), max(lon for _, lon in all_bounds)]],
                     max_zoom=16, padding=(35, 35))
    info = (
        '<div style="position:fixed;top:12px;left:55px;z-index:9999;'
        'background:rgba(255,255,255,.97);border:1px solid #bbb;'
        'padding:11px;border-radius:7px;font:12px Arial;max-width:360px;'
        'box-shadow:0 2px 7px #0003;pointer-events:none">'
        '<b style="font-size:14px">One detected recurrent convoy</b><br>'
        f'Anchor ID: <b>{html.escape(anchor_id)}</b> · {selected_date.date()}<br>'
        f'Recurrence chain: {html.escape(" → ".join(chain))}<br>'
        f'Anchor taxis: {html.escape(", ".join(member_ids))}<br>'
        f'{highlighted} occurrence(s) on this date<br>'
        '<span style="color:#d81b34">Bold red = selected anchor occurrence</span><br>'
        '<span style="color:#cc8300">Orange = earlier detected occurrence</span><br>'
        'Other colors = taxis’ complete observed daily routes<br>'
        '<small>Green dot = first GPS sample · red dot = last.<br>'
        'Toggle taxi layers in the top-right.</small></div>')
    m.get_root().html.add_child(folium.Element(info))
    folium.LayerControl(collapsed=False).add_to(m)

    target = Path(args.output)
    if target.suffix.lower() != '.html':
        target = Path(f'{target}.html')
    target.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(target))
    print(f'Anchor {anchor_id} · members: {member_ids} · recurrence chain: {chain}')
    print(f'Plotted {highlighted} detected recurrence occurrence(s) on {selected_date.date()}')
    print(f'Saved map: {target.resolve()}')
    return m




def recurrent_taxis_map(args, occurrences):
    """Map all detected recurrent occurrences involving the selected taxis.

    Recurrence is established by detector recurrence_edges.csv, not by planted
    labels or mere occurrence membership. Match every selected taxi on *both*
    ends (all), or at least one common selected taxi on both ends (any).
    """
    if not args.input:
        raise ValueError('--input is required to plot full-day GPS tracks.')
    if args.track_gap_seconds <= 0 or args.gap_seconds <= 0 or args.gps_dots < 0:
        raise ValueError('Track/convoy gap must be positive; GPS dots nonnegative.')
    requested = list(dict.fromkeys(str(t) for t in args.taxi_ids))
    if not requested or any(not t.strip() for t in requested):
        raise ValueError('--taxi-ids must contain nonempty taxi IDs.')
    wanted = set(requested)
    edges = read_result_table(args.results, 'recurrence_edges.csv')
    if not {'historical_id', 'later_id'} <= set(edges.columns):
        raise ValueError('recurrence_edges.csv must have historical_id,later_id')

    by_id = occurrences.set_index('convoy_id', drop=False)
    member_sets = {str(row.convoy_id): set(read_objects(row.objects))
                   for row in occurrences.itertuples(index=False)}
    matched_edges = []
    for r in edges.itertuples(index=False):
        old, new = str(r.historical_id), str(r.later_id)
        if old not in member_sets or new not in member_sets:
            raise ValueError(f'Recurrence edge {old} -> {new} refers to missing convoy')
        shared = member_sets[old] & member_sets[new] & wanted
        if (wanted <= shared) if args.match == 'all' else bool(shared):
            matched_edges.append((old, new))
    if not matched_edges:
        raise ValueError('No detector recurrence links contain the requested taxi IDs '
                         f'with --match {args.match}. Check --results and --taxi-ids.')

    # Each recurrence-connected set gets its own map layer, so different
    # recurrent groups can be toggled independently.
    parents = {}
    def root(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    for a, b in matched_edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parents[rb] = ra
    components = defaultdict(set)
    for a, b in matched_edges:
        components[root(a)].update((a, b))
    components = sorted(components.values(),
                        key=lambda component: (min(parse_time(by_id.loc[c]['start_time'])
                                                   for c in component),
                                               min(component)))
    all_selected = set().union(*components)
    earliest = min(parse_time(by_id.loc[c]['start_time']) for c in all_selected)
    day_start = (parse_time(args.date).normalize() if args.date
                 else earliest.normalize())
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    per_component = []
    for component in components:
        day_events = [by_id.loc[c] for c in component
                      if parse_time(by_id.loc[c]['start_time']) <= day_end and
                         parse_time(by_id.loc[c]['end_time']) >= day_start]
        day_events.sort(key=lambda o: (parse_time(o['start_time']), o['convoy_id']))
        if day_events:
            per_component.append((component, day_events))
    if not per_component:
        dates = sorted({parse_time(by_id.loc[c]['start_time']).date().isoformat()
                        for c in all_selected})
        preview = ', '.join(dates[:10])
        raise ValueError(f'No matching recurrent occurrences on {day_start.date()}. '
                         f'Dates with matches include: {preview}')

    data = load_daily_member_tracks(args.input, day_start, day_end, wanted)
    if data.empty:
        raise ValueError('Selected taxis have no GPS samples on the chosen day. '
                         'Pass the same trajectories.csv used by the detector.')

    m = folium.Map(location=[39.9, 116.4], zoom_start=11,
                   tiles=carto_tiles, attr="© OpenStreetMap contributors, © CARTO", control_scale=True, prefer_canvas=True)
    #folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    all_bounds = []
    colors = ['#2563eb', '#16a34a', '#9333ea', '#db2777', '#0d9488',
              '#ca8a04', '#7c3aed', '#0284c7', '#a16207']
    for idx, taxi_id in enumerate(requested):
        taxi = data[data.id == taxi_id]
        if taxi.empty:
            print(f'Warning: no daily GPS records for taxi {taxi_id}')
            continue
        color = colors[idx % len(colors)]
        layer = folium.FeatureGroup(name=f'Taxi {html.escape(taxi_id)} · full-day route',
                                    show=True)
        all_bounds.extend(put_polyline(
            layer, taxi, color, 3.4, 0.8,
            f'Taxi {html.escape(taxi_id)} · full-day GPS route',
            args.track_gap_seconds))
        if args.gps_dots:
            count = min(args.gps_dots, len(taxi))
            indices = (sorted({round(k * (len(taxi)-1) / (count-1)) for k in range(count)})
                       if count > 1 else [0])
            for sample in taxi.iloc[indices].itertuples():
                folium.CircleMarker([float(sample.y), float(sample.x)],
                                    radius=2.8, color='#202020', weight=.7,
                                    fill=True, fill_color='#f33c3c', fill_opacity=.85,
                                    tooltip=f'Taxi {taxi_id} · {sample.time}').add_to(layer)
        for record, label, marker_color in ((taxi.iloc[0], 'first', '#22c55e'),
                                            (taxi.iloc[-1], 'last', '#ef4444')):
            folium.CircleMarker([float(record.y), float(record.x)], radius=6,
                                color='#333', weight=1.5, fill=True,
                                fill_color=marker_color, fill_opacity=1,
                                tooltip=f'Taxi {taxi_id} {label} observation: {record.time}').add_to(layer)
        layer.add_to(m)

    group_colors = ['#df2637', '#f59e0b', '#a21caf', '#0284c7', '#059669',
                    '#b45309', '#7c3aed', '#e11d48']
    plotted = 0
    for idx, (component, day_events) in enumerate(per_component):
        group_color = group_colors[idx % len(group_colors)]
        layer = folium.FeatureGroup(
            name=(f'Recurrent group {idx+1}: {len(day_events)} occurrence(s) today, '
                  f'{len(component)} linked total'), show=True)
        for event in day_events:
            occ_id = str(event['convoy_id'])
            start = parse_time(event['start_time'])
            end = parse_time(event['end_time'])
            occ_members = read_objects(event['objects'])
            included = wanted & set(occ_members)
            part = data.loc[data.id.isin(included) & data.time.between(start, end)]
            if part.empty:
                print(f'Warning: no GPS samples for occurrence {occ_id} at {start}')
                continue
            for taxi_id, taxi in part.groupby('id'):
                all_bounds.extend(put_polyline(
                    layer, taxi, group_color, 8, 0.96,
                    f'Detected recurrent occurrence {occ_id} · Taxi {taxi_id} · {start}–{end}',
                    args.gap_seconds))
            first = part[part.time == part.time.min()]
            coords = [float(first.y.mean()), float(first.x.mean())]
            popup = (f'<b>Detected recurrent convoy occurrence {html.escape(occ_id)}</b><br>'
                     f'{html.escape(str(start))} to {html.escape(str(end))}<br>'
                     f'All convoy members: {html.escape(", ".join(occ_members))}<br>'
                     f'Selected taxis present: {html.escape(", ".join(sorted(included)))}<br>'
                     f'Recurrence component: {idx+1}')
            folium.CircleMarker(coords, radius=8, weight=2, color='#4b1220',
                                fill=True, fill_color=group_color, fill_opacity=.96,
                                popup=folium.Popup(popup, max_width=360),
                                tooltip=f'Recurrent occurrence {occ_id} · {start}').add_to(layer)
            plotted += 1
        layer.add_to(m)
    if plotted == 0:
        raise ValueError('No selected recurrence GPS rows match detector timestamps. '
                         'Check --input points to the exact detector input CSV.')

    if all_bounds:
        m.fit_bounds([[min(lat for lat, _ in all_bounds), min(lon for _, lon in all_bounds)],
                      [max(lat for lat, _ in all_bounds), max(lon for _, lon in all_bounds)]],
                     max_zoom=16, padding=(35, 35))
    summary = (
        '<div style="position:fixed;top:12px;left:55px;z-index:9999;'
        'background:rgba(255,255,255,.97);border:1px solid #bbb;'
        'padding:11px;border-radius:7px;font:12px Arial;max-width:370px;'
        'box-shadow:0 2px 7px #0003;pointer-events:none">'
        '<b style="font-size:14px">All detected recurrent convoys for selected taxis</b><br>'
        f'Taxi IDs: {html.escape(", ".join(requested))}<br>'
        f'Membership matching: <b>{html.escape(args.match)}</b><br>'
        f'Date: {day_start.date()}<br>'
        f'{len(matched_edges)} detected recurrence link(s) across the results<br>'
        f'{len(components)} recurrent component(s) across the results<br>'
        f'{plotted} occurrence(s) plotted on this date<br>'
        'Colored thin lines = selected taxis’ full-day GPS routes<br>'
        'Thick colored lines = detected recurrent convoy sections<br>'
        'Green/red circles = first/last daily observations<br>'
        '<small>Use the layer menu to toggle taxis and recurrence groups.</small>'
        '</div>')
    m.get_root().html.add_child(folium.Element(summary))
    folium.LayerControl(collapsed=False).add_to(m)
    output = Path(args.output)
    if output.suffix.lower() != '.html':
        output = Path(f'{output}.html')
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    print(f'Taxi IDs {requested}, match={args.match}: {len(matched_edges)} recurrence links '
          f'in {len(components)} recurrent components; '
          f'{plotted} distinct detected occurrences on {day_start.date()}.')
    print(f'Saved map: {output.resolve()}')
    return m


# -----------------------------------------------------------------------------
# Direct recurrence-pair mode: show two occurrence IDs, independent colored
# overlays, and the EXACT linked cluster samples used by the detector.
# This mode does not need the original trajectories.csv.
# -----------------------------------------------------------------------------

def load_pair_cluster_details(results_path, ids):
    """Stream saved cluster snapshots for the selected IDs; never load whole JSONL."""
    result_root = Path(results_path)
    cluster_paths = sorted(result_root.rglob('clusters.jsonl'))
    if len(cluster_paths) != 1:
        raise ValueError(f'Expected one clusters.jsonl under {result_root}; found {len(cluster_paths)}.')
    remaining = set(ids)
    found = {}
    with cluster_paths[0].open('r', encoding='utf-8') as f:
        for line in f:
            if not remaining:
                break
            item = json.loads(line)
            key = str(item['cluster_id'])
            if key in remaining:
                found[key] = item
                remaining.remove(key)
    if remaining:
        raise ValueError(f'Missing {len(remaining)} referenced cluster snapshots, e.g. '
                         f'{sorted(remaining)[:10]}. Check that --results contains matching files.')
    return found


def result_projection(results_path):
    paths = sorted(Path(results_path).rglob('run_summary.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected one run_summary.json under {results_path}; found {len(paths)}')
    with paths[0].open('r', encoding='utf-8') as f:
        meta = json.load(f)
    projection = meta.get('projection', {})
    coords = projection.get('coordinates')
    if coords != 'lonlat':
        raise ValueError(f'This map expects detector lonlat projected points; got {coords!r}. '
                         'Original x/y GPS conversion is not available for this projection.')
    for name in ('lon0', 'lat0', 'meters_per_degree_x', 'meters_per_degree_y'):
        if name not in projection or not float(projection[name]):
            raise ValueError(f'run_summary.json is missing valid projection parameter {name}')
    return projection, meta.get('parameters', {})


def cluster_point_latlon(point, projection):
    return (float(projection['lat0']) + float(point[1]) / float(projection['meters_per_degree_y']),
            float(projection['lon0']) + float(point[0]) / float(projection['meters_per_degree_x']))


def split_timed_positions(positions, max_gap_seconds):
    """Yield consecutive position sequences without drawing across data gaps."""
    current = []
    previous = None
    for when, coords in positions:
        if previous is not None and (when - previous).total_seconds() > max_gap_seconds:
            if current:
                yield current
            current = []
        current.append(coords)
        previous = when
    if current:
        yield current


def load_pair_daily_tracks(input_path, wanted_taxi_days):
    """Load every GPS ping on the requested local calendar days in one CSV pass.

    wanted_taxi_days is a set of (taxi_id_string, YYYY-MM-DD_string) pairs.
    This can contain taxis from different recurrence occurrences or different
    calendar days. Never interpolate, truncate, or downsample the route lines.
    """
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Original trajectory CSV not found: {path}. Use --input with the '
            'same id,time,x,y file used to generate these detector results.')
    wanted_taxis = {tid for tid, _ in wanted_taxi_days}
    wanted_days = {day for _, day in wanted_taxi_days}
    wanted_keys = {f'{tid}|{day}' for tid, day in wanted_taxi_days}
    chunks = []
    scanned = 0
    for chunk in pd.read_csv(path, usecols=['id', 'time', 'x', 'y'],
                             dtype={'id': str, 'time': str}, chunksize=250_000):
        scanned += len(chunk)
        # Cheap string filtering before parsing dates and coordinates.
        rough = chunk.loc[chunk['id'].isin(wanted_taxis) &
                          chunk['time'].str[:10].isin(wanted_days)].copy()
        if rough.empty:
            continue
        rough['time'] = pd.to_datetime(rough['time'], errors='coerce')
        rough['x'] = pd.to_numeric(rough['x'], errors='coerce')
        rough['y'] = pd.to_numeric(rough['y'], errors='coerce')
        rough = rough.dropna(subset=['id', 'time', 'x', 'y'])
        rough = rough.loc[rough['x'].between(-180, 180) &
                          rough['y'].between(-90, 90)]
        if rough.empty:
            continue
        keys = rough['id'] + '|' + rough['time'].dt.strftime('%Y-%m-%d')
        rough = rough.loc[keys.isin(wanted_keys)]
        if not rough.empty:
            chunks.append(rough)
    if not chunks:
        raise ValueError('No taxi GPS positions on the convoy dates. Ensure '
                         '--input points to the ORIGINAL detector input CSV.')
    data = pd.concat(chunks, ignore_index=True)
    data = data.sort_values(['id', 'time']).reset_index(drop=True)
    print(f'Scanned {scanned:,} original GPS rows; loaded {len(data):,} daily '
          f'observations for {len(wanted_taxi_days)} taxi/date combinations.')
    return data


def recurrence_pair_map(args):
    """Map an explicitly exported historical->later edge, not matching taxi IDs."""
    if not args.results:
        raise ValueError('--results FOLDER is required for recurrence-pair mode')
    if args.gap_seconds <= 0:
        raise ValueError('--gap-seconds must be positive')

    earlier_id = str(args.historical_id)
    later_id = str(args.later_id)
    if earlier_id == later_id:
        raise ValueError('The historical and later IDs must be different occurrences')

    occ = read_result_table(args.results, 'convoys.csv')
    if 'convoy_id' not in occ or 'cluster_ids' not in occ or 'objects' not in occ:
        raise ValueError('convoys.csv must contain convoy_id,cluster_ids,objects')
    if occ.convoy_id.duplicated().any():
        raise ValueError('Duplicate convoy IDs in convoys.csv')
    by_id = occ.set_index('convoy_id')
    for oid in (earlier_id, later_id):
        if oid not in by_id.index:
            raise ValueError(f'Convoy ID {oid} not found in convoys.csv')

    edges = read_result_table(args.results, 'recurrence_edges.csv')
    if not {'historical_id', 'later_id'} <= set(edges.columns):
        raise ValueError('recurrence_edges.csv needs historical_id,later_id')
    link = edges.loc[(edges['historical_id'] == earlier_id) &
                     (edges['later_id'] == later_id)]
    if link.empty:
        reverse = ((edges['historical_id'] == later_id) &
                   (edges['later_id'] == earlier_id)).any()
        extra = (' They exist in the opposite order; swap the IDs.' if reverse else
                 ' Check recurrence_edges.csv for an actual historical_id,later_id pair.')
        raise ValueError(f'No detector recurrence link {earlier_id} -> {later_id}.{extra}')
    edge = link.iloc[0]
    count = int(edge['match_samples'])
    if count < 1:
        raise ValueError('Recurrence link has an invalid match_samples value')

    details = []
    needed_clusters = set()
    for label, oid, offset_col in (
            ('Historical', earlier_id, 'historical_match_offset'),
            ('Later', later_id, 'later_match_offset')):
        row = by_id.loc[oid]
        cluster_ids = read_objects(row['cluster_ids'])
        offset = int(edge[offset_col])
        if offset < 0 or offset + count > len(cluster_ids):
            raise ValueError(f'Invalid matched cluster window for convoy {oid}: '
                             f'offset {offset}, count {count}, length {len(cluster_ids)}')
        ids = read_objects(row['objects'])
        if not ids:
            raise ValueError(f'Convoy {oid} has no member IDs')
        needed_clusters.update(cluster_ids)
        details.append(dict(label=label, id=oid, row=row, members=ids,
                            clusters=cluster_ids,
                            matches=cluster_ids[offset:offset+count],
                            match_offset=offset))

    # The original trajectory CSV is optional for backwards compatibility.
    # When present, load full calendar-day routes for EVERY member of BOTH
    # occurrences, including taxis with no shared membership.
    pair_focus = args.pair_focus or ('day' if args.input else 'match')
    if pair_focus == 'day' and not args.input:
        raise ValueError('--pair-focus day requires --input ORIGINAL_TRAJECTORIES.csv')
    if args.input and (args.track_gap_seconds <= 0 or args.gps_dots < 0):
        raise ValueError('--track-gap-seconds must be positive and --gps-dots nonnegative')
    taxi_day_keys = set()
    for record in details:
        first_day = parse_time(record['row']['start_time']).normalize()
        last_day = parse_time(record['row']['end_time']).normalize()
        if last_day < first_day:
            raise ValueError(f'Convoy {record["id"]} has an end time before its start')
        for day in pd.date_range(first_day, last_day, freq='D'):
            for taxi_id in record['members']:
                taxi_day_keys.add((taxi_id, day.strftime('%Y-%m-%d')))
    daily_data = (load_pair_daily_tracks(args.input, taxi_day_keys)
                  if args.input else None)

    projection, parameters = result_projection(args.results)
    snapshots = load_pair_cluster_details(args.results, needed_clusters)
    colors = {'Historical': '#1160dc', 'Later': '#ee780c'}

    m = folium.Map(location=[39.9, 116.4], zoom_start=15,
                   tiles=carto_tiles,
                   attr='© OpenStreetMap contributors, © CARTO',
                   control_scale=True, prefer_canvas=True)
    all_coords, matching_coords, daily_coords = [], [], []
    daily_layers = 0
    if daily_data is not None:
        # Each member taxi gets its own toggle for each day it participated.
        # Keep complete daily tracks behind the bold, color-coded match traces.
        taxi_colors = ['#0d9488', '#9333ea', '#db2777', '#64748b',
                       '#16a34a', '#a21caf', '#0369a1', '#b45309',
                       '#4f46e5', '#be123c', '#15803d', '#475569']
        unique_taxis = sorted({tid for tid, _ in taxi_day_keys},
                             key=lambda value: (not value.isdigit(),
                                                int(value) if value.isdigit() else value))
        taxi_color_by_id = {tid: taxi_colors[i % len(taxi_colors)]
                            for i, tid in enumerate(unique_taxis)}
        for taxi_id, day in sorted(taxi_day_keys, key=lambda key: (key[1], key[0])):
            day_start = parse_time(day)
            day_end = day_start + pd.Timedelta(days=1)
            track = daily_data.loc[(daily_data['id'] == taxi_id) &
                                   daily_data['time'].ge(day_start) &
                                   daily_data['time'].lt(day_end)]
            if track.empty:
                print(f'Warning: no original daily GPS rows for taxi {taxi_id} on {day}')
                continue
            color = taxi_color_by_id[taxi_id]
            layer = folium.FeatureGroup(
                name=f'Taxi {html.escape(taxi_id)} · full-day route {day}', show=True)
            coords = put_polyline(layer, track, color, 2.4, .48,
                                  f'Taxi {html.escape(taxi_id)} · full-day GPS · {day}',
                                  args.track_gap_seconds)
            daily_coords.extend(coords)
            if args.gps_dots:
                dots = min(len(track), args.gps_dots)
                choices = (sorted({round(j * (len(track)-1) / (dots-1))
                                   for j in range(dots)}) if dots > 1 else [0])
                for pt in track.iloc[choices].itertuples():
                    folium.CircleMarker((float(pt.y), float(pt.x)),
                                        radius=2, color=color, weight=.5,
                                        fill=True, fill_color=color, fill_opacity=.65,
                                        tooltip=f'Taxi {taxi_id} · {pt.time}').add_to(layer)
            first, last = track.iloc[0], track.iloc[-1]
            for pt, name, mark_color in ((first, 'START', '#16a34a'),
                                         (last, 'END', '#dc2626')):
                folium.CircleMarker((float(pt.y), float(pt.x)), radius=5.5,
                                    weight=1.5, color='white', fill=True,
                                    fill_color=mark_color, fill_opacity=.98,
                                    tooltip=f'Taxi {taxi_id} {name} · {pt.time}').add_to(layer)
            layer.add_to(m)
            daily_layers += 1
        if not daily_coords:
            raise ValueError('No daily GPS tracks were found for any convoy members')
        print(f'Drew {daily_layers} independently toggleable full-day taxi tracks')

    for record in details:
        label, oid = record['label'], record['id']
        color = colors[label]
        start = parse_time(record['row']['start_time'])
        end = parse_time(record['row']['end_time'])
        layer = folium.FeatureGroup(
            name=f'{label} convoy {oid} · {start.strftime("%Y-%m-%d %H:%M:%S")}',
            show=True)
        is_match = set(record['matches'])
        member_tracks = defaultdict(list)
        match_tracks = defaultdict(list)
        # Reconstruct the actual matched DBSCAN cluster points for each member.
        # We show the complete detected occurrence as a thinner route, and the
        # subsection actually participating in THIS recurrence edge as a thicker route.
        for cluster_id in record['clusters']:
            cluster = snapshots[cluster_id]
            when = parse_time(cluster['time'])
            objects = [str(obj) for obj in cluster['objects']]
            points = cluster['points_m']
            if len(objects) != len(points):
                raise ValueError(f'Cluster {cluster_id}: points_m and objects length mismatch')
            for member, xy in zip(objects, points):
                if member not in record['members']:
                    continue
                pos = cluster_point_latlon(xy, projection)
                member_tracks[member].append((when, pos))
                all_coords.append(pos)
                if cluster_id in is_match:
                    match_tracks[member].append((when, pos))
                    matching_coords.append(pos)

        # Full occurrence, muted but present so the user can see that most of a
        # long occurrence did not necessarily match the earlier/later convoy.
        for member in record['members']:
            track = member_tracks.get(member, [])
            for part in split_timed_positions(track, args.gap_seconds):
                if len(part) > 1:
                    folium.PolyLine(part, color=color, opacity=.46, weight=4,
                                   tooltip=f'{label} convoy {oid} · taxi {member} · whole occurrence').add_to(layer)
                elif part:
                    folium.CircleMarker(part[0], radius=2.5, color=color, fill=True,
                                        fill_color=color, fill_opacity=.65,
                                        tooltip=f'Convoy {oid} taxi {member}').add_to(layer)

        # Matched section is a distinct, vivid trace, and each matched sample
        # gets a tooltip with its true detector timestamp (no inferred timing).
        for member in record['members']:
            match = match_tracks.get(member, [])
            for part in split_timed_positions(match, args.gap_seconds):
                if len(part) > 1:
                    folium.PolyLine(part, color=color, opacity=1, weight=8,
                                   tooltip=f'{label} convoy {oid} · taxi {member} · matched sequence').add_to(layer)
            for j, (when, pos) in enumerate(match):
                folium.CircleMarker(pos, radius=4.5, color='white', weight=1.3,
                                    fill=True, fill_color=color, fill_opacity=.95,
                                    tooltip=f'{label} {oid} · taxi {member} · matched {when}',
                                    popup=folium.Popup(
                                        f'<b>{html.escape(label)} convoy {html.escape(oid)}</b><br>'
                                        f'Taxi {html.escape(member)}<br>'
                                        f'Matching timestamp: {html.escape(str(when))}<br>'
                                        f'Sample {j+1} of {len(match)}', max_width=300)).add_to(layer)

        # A central marker at the beginning of the MATCHED part (not the entire
        # convoy) lets the audience identify where recurrence happened.
        first_cluster = snapshots[record['matches'][0]]
        first_points = [cluster_point_latlon(p, projection)
                        for member, p in zip(first_cluster['objects'], first_cluster['points_m'])
                        if str(member) in record['members']]
        if not first_points:
            raise ValueError(f'No convoy members found in matched cluster for {oid}')
        center = (sum(p[0] for p in first_points) / len(first_points),
                  sum(p[1] for p in first_points) / len(first_points))
        tip = (f'{label} convoy {oid} | {start} to {end} | '
               f'{len(record["members"])} member taxis | matched {count} snapshots')
        folium.CircleMarker(center, radius=11, color='#151515', weight=2.5,
                            fill=True, fill_color=color, fill_opacity=.89,
                            popup=folium.Popup(
                                f'<b>{html.escape(label)} convoy {html.escape(oid)}</b><br>'
                                f'Occurrence: {html.escape(str(start))} – {html.escape(str(end))}<br>'
                                f'Matched snapshots: {count}<br>'
                                f'Taxis: {html.escape(", ".join(record["members"]))}',
                                max_width=390), tooltip=tip).add_to(layer)
        layer.add_to(m)
        print(f'{label} convoy {oid}: {len(record["members"])} taxis; '
              f'{len(record["clusters"])} full cluster snapshots; '
              f'{len(record["matches"])} matching snapshots')

    frame_coords = ({'match': matching_coords, 'all': all_coords,
                     'day': daily_coords}[pair_focus])
    if frame_coords:
        minlat = min(p[0] for p in frame_coords)
        maxlat = max(p[0] for p in frame_coords)
        minlon = min(p[1] for p in frame_coords)
        maxlon = max(p[1] for p in frame_coords)
        # Add a small visible margin for tiny matching windows.
        lat_pad = max((maxlat - minlat) * .13, .00022)
        lon_pad = max((maxlon - minlon) * .13, .00028)
        m.fit_bounds([[minlat-lat_pad, minlon-lon_pad],
                      [maxlat+lat_pad, maxlon+lon_pad]],
                     max_zoom=18, padding=(30, 30))

    before, after = details
    gap_seconds = (parse_time(after['row']['start_time']) -
                   parse_time(before['row']['start_time'])).total_seconds()
    shared = sorted(set(before['members']) & set(after['members']))
    limit = parameters.get('delta_m')
    limit_label = (f'{float(limit):g} m' if limit is not None else 'not provided')
    legend = (
        '<div style="position:fixed;top:12px;left:55px;z-index:9999;'
        'background:rgba(255,255,255,.97);border:1px solid #aaa;'
        'padding:12px 13px;border-radius:8px;font:12px/1.45 Arial;'
        'width:340px;max-width:calc(100vw - 110px);box-shadow:0 2px 8px #0003;pointer-events:none">'
        '<b style="font-size:15px">Verified detector recurrence link</b><br>'
        f'<span style="color:{colors["Historical"]}"><b>● Historical convoy {html.escape(earlier_id)}</b></span>'
        f' · {html.escape(str(parse_time(before["row"]["start_time"]))) }<br>'
        f'Members: {html.escape(", ".join(before["members"]))}<br>'
        f'<span style="color:{colors["Later"]}"><b>● Later convoy {html.escape(later_id)}</b></span>'
        f' · {html.escape(str(parse_time(after["row"]["start_time"]))) }<br>'
        f'Members: {html.escape(", ".join(after["members"]))}<br>'
        f'Shared taxi IDs: {html.escape(", ".join(shared)) if shared else "none (allowed)"}<br>'
        f'Start-time gap: {gap_seconds:,.0f} seconds<br>'
        f'Matched consecutive cluster snapshots: {count}<br>'
        f'Spatial threshold configured: {html.escape(limit_label)}<br>'
        '<b>Thick lines/dots:</b> matched cluster positions<br>'
        '<b>Medium lines:</b> remainder of detected convoy occurrence<br>'
        + (f'<b>Thin colored routes:</b> {daily_layers} complete taxi/day tracks, each toggleable<br>'
           'Green/red dots on daily routes = first/last GPS observation<br>'
           if daily_data is not None else '<b>Daily GPS routes:</b> add --input original.csv<br>')
        + f'<small>Use layer controls to toggle each convoy or daily taxi route. Initial view: {html.escape(pair_focus)}.</small>'
        '</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    output = Path(args.output)
    if output.suffix.lower() != '.html':
        output = Path(str(output) + '.html')
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    print(f'Confirmed detector recurrence edge: {earlier_id} -> {later_id}; '
          f'{count} matched snapshots; {gap_seconds:g}s between starts')
    print(f'Saved map: {output.resolve()}')
    return m


def main():
    args = parse_args()
    try:
        pair_mode = args.historical_id is not None or args.later_id is not None
        if pair_mode:
            if args.historical_id is None or args.later_id is None:
                raise ValueError('Supply BOTH --historical-id and --later-id.')
            if args.taxi_ids or args.anchor_id is not None or args.list_anchors:
                raise ValueError('Choose --historical-id/--later-id OR --taxi-ids/--anchor-id/--list-anchors.')
            recurrence_pair_map(args)
        elif args.list_anchors or args.anchor_id is not None or args.taxi_ids:
            if args.taxi_ids and args.anchor_id is not None:
                raise ValueError('Choose --taxi-ids OR --anchor-id, not both.')
            if not args.results:
                raise ValueError('--results folder is required for --anchor-id, --taxi-ids or --list-anchors')
            occ, recur = detector_tables(args.results)
            if args.list_anchors:
                print_recurrent_anchors(occ, recur)
            elif args.taxi_ids:
                recurrent_taxis_map(args, occ)
            else:
                one_recurrent_convoy_map(args, occ, recur)
        else:
            if not args.input or not args.convoys:
                raise ValueError('Use --historical-id and --later-id for a recurrence pair; '
                                 'or --input and --convoys for the original multi-convoy map.')
            make_map(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f'Error: {exc}') from exc


if __name__ == '__main__':
    main()
