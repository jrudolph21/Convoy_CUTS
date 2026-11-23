#!/usr/bin/env python3
"""Create an interactive Folium map of one taxi's one-day trajectory.

Expected CSV columns are id,time,x,y, where x is longitude and y is latitude.
A header is optional. If --date is omitted, the script requires the selected
taxi to have points from exactly one calendar day in the input file.
"""

import argparse
import csv
import re
from datetime import date, datetime
from pathlib import Path

import folium

CARTO_API_KEY = "cb1_3wq7_1_44372b8a982bea61dc3eedfd"

carto_tiles = (
    "https://basemaps.cartocdn.com/rastertiles/"
    "light_all/{z}/{x}/{y}.png"
    f"?key={CARTO_API_KEY}"
)

def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
    raise ValueError(f"Unrecognized timestamp: {value!r}")


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from error


def read_taxi_points(input_path: Path, taxi_id: str, selected_day: date | None):
    """Read the chosen taxi and its numbered split IDs, optionally for one date."""
    points = []
    taxi_id_pattern = re.compile(rf"^{re.escape(str(taxi_id))}(?:\.\d+)?$")
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        column_map = {"id": 0, "time": 1, "x": 2, "y": 3}
        first_nonempty_row = True

        for row_number, row in enumerate(reader, start=1):
            if not row or not any(value.strip() for value in row):
                continue

            if first_nonempty_row:
                first_nonempty_row = False
                names = [value.strip().lower() for value in row]
                if all(name in names for name in ("id", "time", "x", "y")):
                    column_map = {name: names.index(name) for name in ("id", "time", "x", "y")}
                    continue

            try:
                taxi_value = row[column_map["id"]].strip()
                if not taxi_id_pattern.fullmatch(taxi_value):
                    continue
                timestamp = parse_timestamp(row[column_map["time"]])
                longitude = float(row[column_map["x"]])
                latitude = float(row[column_map["y"]])
            except (IndexError, ValueError) as error:
                raise ValueError(f"Could not parse CSV row {row_number}: {row!r}\n{error}") from error

            if selected_day is not None and timestamp.date() != selected_day:
                continue
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError(
                    f"Invalid coordinates on CSV row {row_number}: x={longitude}, y={latitude}. "
                    "Expected x=longitude and y=latitude."
                )
            points.append((timestamp, latitude, longitude, taxi_value))

    points.sort(key=lambda point: (point[3], point[0]))
    return points


def make_map(points, taxi_id: str, output_path: Path):
    segments = {}
    for timestamp, latitude, longitude, segment_id in points:
        segments.setdefault(segment_id, []).append((timestamp, latitude, longitude))

    all_points = [point for segment in segments.values() for point in segment]
    all_timestamps = [point[0] for point in all_points]
    all_coordinates = [(point[1], point[2]) for point in all_points]
    day = all_timestamps[0].date()

    center = [
        sum(latitude for latitude, _ in all_coordinates) / len(all_coordinates),
        sum(longitude for _, longitude in all_coordinates) / len(all_coordinates),
    ]
    trajectory_map = folium.Map(location=center, zoom_start=13, tiles=carto_tiles, attr="© OpenStreetMap contributors, © CARTO", control_scale=True)

    for segment_id, segment in sorted(segments.items()):
        timestamps = [point[0] for point in segment]
        coordinates = [(point[1], point[2]) for point in segment]
        feature_group = folium.FeatureGroup(name=f"Taxi {segment_id}")

        if len(coordinates) > 1:
            folium.PolyLine(
                coordinates,
                color="#1769aa",
                weight=4,
                opacity=0.85,
                tooltip=f"Taxi {segment_id} trajectory on {day.isoformat()}",
                smooth_factor=1,
            ).add_to(feature_group)

        # Add a small dot at each five-minute clock mark.
        for timestamp, coordinate in zip(timestamps, coordinates):
            if timestamp.minute % 5 == 0 and timestamp.second == 0:
                folium.CircleMarker(
                    coordinate,
                    radius=2,
                    color="#222222",
                    weight=1,
                    fill=True,
                    fill_color="#ffffff",
                    fill_opacity=0.9,
                    tooltip=f"Taxi {segment_id}: {timestamp.strftime('%H:%M:%S')}",
                ).add_to(feature_group)

        start_time = timestamps[0].strftime("%Y-%m-%d %H:%M:%S")
        end_time = timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")
        folium.CircleMarker(
            coordinates[0],
            radius=7,
            color="#16803c",
            fill=True,
            fill_opacity=1,
            tooltip=f"Taxi {segment_id} start: {start_time}",
        ).add_to(feature_group)
        folium.CircleMarker(
            coordinates[-1],
            radius=7,
            color="#c62828",
            fill=True,
            fill_opacity=1,
            tooltip=f"Taxi {segment_id} end: {end_time}",
        ).add_to(feature_group)
        feature_group.add_to(trajectory_map)

    trajectory_map.fit_bounds(
        [
            [min(lat for lat, _ in all_coordinates), min(lon for _, lon in all_coordinates)],
            [max(lat for lat, _ in all_coordinates), max(lon for _, lon in all_coordinates)],
        ]
    )
    folium.LayerControl(collapsed=False).add_to(trajectory_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_map.save(str(output_path))
    return day, min(all_timestamps), max(all_timestamps)


def main():
    parser = argparse.ArgumentParser(
        description="Map the full one-day trajectory of a taxi from an interpolated CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV file with id,time,x,y columns")
    parser.add_argument(
        "--taxi-id", default="1",
        help="Base taxi ID to map, including numbered split IDs such as 1.01 (default: 1)",
    )
    parser.add_argument("--date", type=parse_date, help="Date to map, in YYYY-MM-DD format")
    parser.add_argument("--output", type=Path, help="HTML output path")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")

    points = read_taxi_points(args.input, args.taxi_id, args.date)
    if not points:
        date_text = f" on {args.date.isoformat()}" if args.date else ""
        parser.error(f"no rows found for taxi ID {args.taxi_id}{date_text}")

    unique_days = {point[0].date() for point in points}
    if args.date is None and len(unique_days) != 1:
        parser.error(
            "the taxi has points on multiple dates; pass --date YYYY-MM-DD to select one day"
        )

    day = points[0][0].date()
    output_path = args.output or Path(f"taxi_{args.taxi_id}_{day.isoformat()}.html")
    day, start_time, end_time = make_map(points, args.taxi_id, output_path)
    segment_count = len({point[3] for point in points})
    print(
        f"Mapped {len(points):,} points across {segment_count} segment(s) for taxi "
        f"{args.taxi_id} on {day.isoformat()} ({start_time} to {end_time}) to {output_path}"
    )


if __name__ == "__main__":
    main()
