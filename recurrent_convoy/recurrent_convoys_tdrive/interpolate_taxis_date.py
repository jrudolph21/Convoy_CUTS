#!/usr/bin/env python3
"""Interpolate taxi locations at one-second intervals for a selected calendar day.

Input CSV columns: id,time,x,y (a header is optional). Output is a headered CSV
with the same columns, containing timestamps in the selected day only.
"""
import argparse
import csv
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def read_trajectories(input_path: Path):
    """Read points grouped by taxi ID; accepts an optional CSV header."""
    points_by_taxi = defaultdict(list)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        first_row = True
        for row in reader:
            if not row or len(row) < 4:
                continue
            values = [value.strip() for value in row[:4]]
            if first_row and [value.lower() for value in values] == ["id", "time", "x", "y"]:
                first_row = False
                continue
            first_row = False
            taxi_id, timestamp, x, y = values
            try:
                point = (datetime.strptime(timestamp, DATETIME_FORMAT), float(x), float(y))
            except ValueError as error:
                raise ValueError(f"Invalid input row: {row!r}") from error
            points_by_taxi[taxi_id].append(point)
    return points_by_taxi


def interpolate_day(input_path: Path, output_path: Path, selected_day: date) -> int:
    """Write second-by-second interpolated locations for selected_day only."""
    day_start = datetime.combine(selected_day, time.min)
    day_end = day_start + timedelta(days=1)
    points_by_taxi = read_trajectories(input_path)
    row_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["id", "time", "x", "y"])

        # Keep memory use bounded by writing one taxi at a time. Rows remain
        # chronological within each taxi; taxi groups follow first-seen input order.
        for taxi_id, points in points_by_taxi.items():
            points.sort(key=lambda point: point[0])
            unique_points = []
            last_timestamp = None
            for point in points:
                if point[0] != last_timestamp:
                    unique_points.append(point)
                    last_timestamp = point[0]

            if len(unique_points) == 1:
                timestamp, x, y = unique_points[0]
                if day_start <= timestamp < day_end:
                    writer.writerow([taxi_id, timestamp.strftime(DATETIME_FORMAT), f"{x:.5f}", f"{y:.5f}"])
                    row_count += 1
                continue

            for index in range(len(unique_points) - 1):
                t1, x1, y1 = unique_points[index]
                t2, x2, y2 = unique_points[index + 1]
                if t2 <= day_start or t1 >= day_end or t2 <= t1:
                    continue

                interval_start = max(t1, day_start)
                interval_end = min(t2, day_end)
                total_seconds = (t2 - t1).total_seconds()
                seconds = int((interval_end - interval_start).total_seconds())
                for offset in range(seconds):
                    timestamp = interval_start + timedelta(seconds=offset)
                    ratio = (timestamp - t1).total_seconds() / total_seconds
                    x = x1 + (x2 - x1) * ratio
                    y = y1 + (y2 - y1) * ratio
                    writer.writerow([taxi_id, timestamp.strftime(DATETIME_FORMAT), f"{x:.5f}", f"{y:.5f}"])
                    row_count += 1

            # Include the final source point if it falls inside the selected day.
            timestamp, x, y = unique_points[-1]
            if day_start <= timestamp < day_end:
                writer.writerow([taxi_id, timestamp.strftime(DATETIME_FORMAT), f"{x:.5f}", f"{y:.5f}"])
                row_count += 1

    return row_count


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate taxi trajectories at one-second intervals for one day."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with id,time,x,y columns")
    parser.add_argument("--date", required=True, help="Day to export, in YYYY-MM-DD format")
    parser.add_argument("--output", type=Path, help="Output CSV path (default: interpolated_YYYY-MM-DD.csv)")
    args = parser.parse_args()

    try:
        selected_day = date.fromisoformat(args.date)
    except ValueError as error:
        parser.error(f"--date must use YYYY-MM-DD format: {error}")

    output_path = args.output or Path(f"interpolated_{selected_day.isoformat()}.csv")
    count = interpolate_day(args.input, output_path, selected_day)
    print(f"Wrote {count:,} interpolated rows for {selected_day.isoformat()} to {output_path}")


if __name__ == "__main__":
    main()
