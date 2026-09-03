import argparse
import ast
import pandas as pd
import folium


def parse_objects(value):
    """
    Parse convoy object IDs from the saved CuTS CSV.

    Handles formats such as:
        {110, 205, 310}
        {np.int64(110), np.int64(205), np.int64(310)}
        [110, 205, 310]
        110,205,310
        110
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Extract integer IDs directly.
    # This handles np.int64(...) and normal integer formats.
    import re

    matches = re.findall(r"[-+]?\d+", text)

    if matches:
        return list(dict.fromkeys(int(x) for x in matches))

    # Fallback for normal Python literals
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (set, list, tuple)):
            result = []

            for x in parsed:
                try:
                    result.append(int(x))
                except (TypeError, ValueError):
                    continue

            return list(dict.fromkeys(result))

        return [int(parsed)]

    except (ValueError, SyntaxError, TypeError):
        return []


def load_convoys(convoy_csv):
    df = pd.read_csv(convoy_csv, parse_dates=["start_time", "end_time"])

    required = {"objects", "start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Convoy CSV is missing required columns: {sorted(missing)}"
        )

    convoys = []
    for _, row in df.iterrows():
        objects = parse_objects(row["objects"])
        if not objects:
            continue

        convoys.append(
            {
                "convoy_id": int(row["convoy_id"]) if "convoy_id" in df.columns else len(convoys) + 1,
                "objects": set(objects),
                "start_time": pd.Timestamp(row["start_time"]),
                "end_time": pd.Timestamp(row["end_time"]),
            }
        )

    return convoys


def interpolate_trajectory(
    taxi_df,
    start_time,
    end_time,
    freq_minutes=10
):
    """
    Reconstruct the taxi trajectory over the convoy interval.

    Important:
    We keep observations immediately BEFORE and AFTER the convoy
    interval so interpolation can reproduce virtual positions at
    missing timestamps inside the interval.
    """

    times = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{freq_minutes}min",
    )

    taxi_df = taxi_df[
        ["time", "x", "y"]
    ].dropna().sort_values("time")

    if taxi_df.empty:
        return pd.DataFrame(columns=["time", "x", "y"])

    # ----------------------------------------------------------
    # Keep one observation immediately before the window
    # ----------------------------------------------------------
    before = taxi_df[taxi_df["time"] < start_time]

    before_point = (
        before.iloc[[-1]]
        if not before.empty
        else taxi_df.iloc[0:0]
    )

    # ----------------------------------------------------------
    # Keep one observation immediately after the window
    # ----------------------------------------------------------
    after = taxi_df[taxi_df["time"] > end_time]

    after_point = (
        after.iloc[[0]]
        if not after.empty
        else taxi_df.iloc[0:0]
    )

    # ----------------------------------------------------------
    # Points inside the convoy interval
    # ----------------------------------------------------------
    inside = taxi_df[
        (taxi_df["time"] >= start_time)
        & (taxi_df["time"] <= end_time)
    ]

    # Combine everything needed for interpolation
    interpolation_df = pd.concat(
        [
            before_point,
            inside,
            after_point,
        ],
        ignore_index=True,
    )

    interpolation_df = (
        interpolation_df
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .set_index("time")
    )

    if interpolation_df.empty:
        return pd.DataFrame(columns=["time", "x", "y"])

    # ----------------------------------------------------------
    # Add the desired convoy timestamps
    # ----------------------------------------------------------
    full_index = (
        interpolation_df.index
        .union(times)
        .sort_values()
    )

    interpolation_df = interpolation_df.reindex(full_index)

    # ----------------------------------------------------------
    # Time-based interpolation
    # ----------------------------------------------------------
    interpolation_df[["x", "y"]] = (
        interpolation_df[["x", "y"]]
        .interpolate(
            method="time",
            limit_area="inside"
        )
    )

    # ----------------------------------------------------------
    # Return ONLY timestamps in the convoy interval
    # ----------------------------------------------------------
    result = (
        interpolation_df
        .reindex(times)
        .dropna(subset=["x", "y"])
        .reset_index()
        .rename(columns={"index": "time"})
    )

    return result


def save_map(
    trajectory_csv,
    convoy_csv,
    output_html,
    freq_minutes=10,
    zoom_start=12,
):
    print("[Map] Loading trajectory data...")
    df = pd.read_csv(
        trajectory_csv,
        parse_dates=["time"],
    )

    print("[Map] Loading existing CuTS convoy results...")
    convoys = load_convoys(convoy_csv)

    if not convoys:
        raise RuntimeError("No convoys found in the convoy CSV.")

    active_ids = sorted(
        {taxi_id for convoy in convoys for taxi_id in convoy["objects"]}
    )

    active_df = df[df["id"].isin(active_ids)].copy()

    if active_df.empty:
        raise RuntimeError(
            "None of the convoy taxi IDs were found in the trajectory CSV."
        )

    center = [
        float(active_df["y"].mean()),
        float(active_df["x"].mean()),
    ]

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "cadetblue",
        "darkgreen",
        "pink",
        "black",
    ]

    plotted = 0
    missing = []

    for index, convoy in enumerate(convoys):
        convoy_id = convoy["convoy_id"]
        color = colors[index % len(colors)]

        layer = folium.FeatureGroup(
            name=(
                f"Convoy {convoy_id} | "
                f"size={len(convoy['objects'])} | "
                f"{convoy['start_time']:%H:%M}-"
                f"{convoy['end_time']:%H:%M}"
            )
        )

        for taxi_id in sorted(convoy["objects"]):
            taxi_raw = df[df["id"] == taxi_id]

            taxi = interpolate_trajectory(
                taxi_raw,
                convoy["start_time"],
                convoy["end_time"],
                freq_minutes=freq_minutes,
            )

            route = list(
                zip(
                    taxi["y"].astype(float),
                    taxi["x"].astype(float),
                )
            )

            if len(route) < 2:
                missing.append(
                    (
                        convoy_id,
                        taxi_id,
                        len(route),
                    )
                )

                if len(route) == 1:
                    folium.CircleMarker(
                        location=route[0],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9,
                        tooltip=(
                            f"Convoy {convoy_id} | Taxi {taxi_id} | "
                            "single plotted point"
                        ),
                    ).add_to(layer)

                continue

            folium.PolyLine(
                route,
                color=color,
                weight=5,
                opacity=0.8,
                tooltip=f"Convoy {convoy_id} | Taxi {taxi_id}",
            ).add_to(layer)

            folium.CircleMarker(
                location=route[0],
                radius=5,
                color=color,
                fill=True,
                fill_color="white",
                fill_opacity=1.0,
                tooltip=f"Convoy {convoy_id} | Taxi {taxi_id} start",
            ).add_to(layer)

            folium.CircleMarker(
                location=route[-1],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                tooltip=f"Convoy {convoy_id} | Taxi {taxi_id} end",
            ).add_to(layer)

            plotted += 1

        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_html)

    print(f"[Map] Convoys loaded: {len(convoys)}")
    print(f"[Map] Taxi trajectories plotted: {plotted}")
    print(f"[Map] Saved: {output_html}")

    if missing:
        print(
            f"[Map] {len(missing)} convoy/taxi entries had fewer than "
            "2 plottable points:"
        )
        for convoy_id, taxi_id, npoints in missing[:25]:
            print(
                f"    Convoy {convoy_id}, Taxi {taxi_id}: "
                f"{npoints} plotted point(s)"
            )

        if len(missing) > 25:
            print(f"    ... and {len(missing) - 25} more")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize already-discovered CuTS convoys without rerunning CuTS."
    )

    parser.add_argument(
        "--trajectory-csv",
        default="tdrive_processed.csv",
        help="Preprocessed T-drive trajectory CSV.",
    )

    parser.add_argument(
        "--convoy-csv",
        required=True,
        help="Existing CuTS convoy results CSV.",
    )

    parser.add_argument(
        "--output",
        default="cuts_existing_results_map.html",
        help="Output Folium HTML map.",
    )

    parser.add_argument(
        "--freq-minutes",
        type=int,
        default=10,
        help="Time-grid spacing used for interpolation.",
    )

    parser.add_argument(
        "--zoom-start",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    save_map(
        trajectory_csv=args.trajectory_csv,
        convoy_csv=args.convoy_csv,
        output_html=args.output,
        freq_minutes=args.freq_minutes,
        zoom_start=args.zoom_start,
    )


if __name__ == "__main__":
    main()
