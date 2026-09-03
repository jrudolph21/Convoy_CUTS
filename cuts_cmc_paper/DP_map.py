import pandas as pd
import numpy as np
import folium

def latlon_to_meters(lats, lons):
    """Converts Lat/Lon coordinates into local Cartesian meters using equirectangular projection."""
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    
    lat0 = np.mean(lat_rad)
    lon0 = np.mean(lon_rad)
    
    R = 6371000.0  # Earth's radius in meters
    x = R * (lon_rad - lon0) * np.cos(lat0)
    y = R * (lat_rad - lat0)
    return np.column_stack((x, y))

def perpendicular_distance(pts, p1, p2):
    """Calculates perpendicular distance from point array `pts` to segment line (p1, p2) in meters."""
    if np.all(p1 == p2):
        return np.linalg.norm(pts - p1, axis=1)
    
    num = np.abs((p2[1] - p1[1]) * pts[:, 0] - (p2[0] - p1[0]) * pts[:, 1] + p2[0] * p1[1] - p2[1] * p1[0])
    den = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    return num / den

def dp_simplify_mask(coords, epsilon):
    """Recursive Douglas-Peucker algorithm returning a boolean mask of retained points."""
    if len(coords) <= 2:
        return np.ones(len(coords), dtype=bool)

    p1, p2 = coords[0], coords[-1]
    dists = perpendicular_distance(coords[1:-1], p1, p2)

    if len(dists) == 0:
        return np.ones(len(coords), dtype=bool)

    dmax = np.max(dists)
    index = np.argmax(dists) + 1  # Offset by 1 for slicing

    if dmax > epsilon:
        rec1 = dp_simplify_mask(coords[:index + 1], epsilon)
        rec2 = dp_simplify_mask(coords[index:], epsilon)
        return np.concatenate((rec1[:-1], rec2))
    else:
        mask = np.zeros(len(coords), dtype=bool)
        mask[0], mask[-1] = True, True
        return mask

def simplify_trajectories_dp(df, tolerance_meters=31.5):
    """Applies Douglas-Peucker trajectory simplification grouped by taxi ID."""
    simplified_dfs = []
    
    for taxi_id, group in df.groupby('id'):
        group_sorted = group.sort_values(by='time').copy()
        
        if len(group_sorted) <= 2:
            simplified_dfs.append(group_sorted)
            continue

        coords_m = latlon_to_meters(group_sorted['y'].values, group_sorted['x'].values)
        keep_mask = dp_simplify_mask(coords_m, epsilon=tolerance_meters)
        simplified_dfs.append(group_sorted.iloc[keep_mask])

    return pd.concat(simplified_dfs, ignore_index=True)

def visualize_comparison_traj(df_orig, df_dp, file="traj_map_comparison"):
    """
    Plots both original and DP-simplified trajectories on a single interactive map.
    - Original paths: Dashed, lower opacity.
    - DP Simplified paths: Solid lines with vertex markers at retained points.
    """
    if df_orig is None or df_orig.empty:
        print("No data available to visualize.")
        return

    center_lat = df_orig['y'].mean()
    center_lon = df_orig['x'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
    
    # Add floating HTML Title Card with Overall Window Start & End Times
    fmt_start = pd.to_datetime(start_time).strftime('%Y-%m-%d %H:%M') if start_time else "N/A"
    fmt_end = pd.to_datetime(end_time).strftime('%Y-%m-%d %H:%M') if end_time else "N/A"

    title_html = f'''
            <div style="position: fixed; 
                        top: 10px; left: 50px; width: 340px; 
                        background-color: white; border:2px solid #ccc; z-index:9999; 
                        font-family: Arial, sans-serif; font-size:13px; font-weight: bold; 
                        text-align: center; padding: 8px; border-radius: 6px;
                        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
                Beijing Taxi Convoy Map<br>
                <span style="font-size:11px; font-weight: normal; color: #444;">
                    Analysis Window: <b>{fmt_start}</b> to <b>{fmt_end}</b>
                </span>
            </div>
            '''
    m.get_root().html.add_child(folium.Element(title_html))
        
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'pink', 'black']

    # Master Layer Groups for toggling all original vs simplified at once
    group_orig_master = folium.FeatureGroup(name=f"Original Trajectories ({len(df_orig)} total pts)")
    group_dp_master = folium.FeatureGroup(name=f"DP Simplified Trajectories ({len(df_dp)} total pts)")

    unique_taxis = df_orig['id'].unique()

    for i, taxi_id in enumerate(unique_taxis):
        traj_color = colors[i % len(colors)]

        # --- 1. Plot Original Trajectory ---
        orig_data = df_orig[df_orig['id'] == taxi_id].sort_values(by='time')
        orig_route = list(zip(orig_data['y'], orig_data['x']))

        if orig_route:
            folium.PolyLine(
                orig_route,
                weight=3,
                color=traj_color,
                dash_array='6, 6',  # Dashed line for original raw data
                opacity=0.45,
                tooltip=f"Original Taxi {taxi_id} ({len(orig_route)} pts)"
            ).add_to(group_orig_master)

        # --- 2. Plot DP Simplified Trajectory ---
        dp_data = df_dp[df_dp['id'] == taxi_id].sort_values(by='time')
        dp_route = list(zip(dp_data['y'], dp_data['x']))

        if dp_route:
            folium.PolyLine(
                dp_route,
                weight=4,
                color=traj_color,
                opacity=0.9,
                tooltip=f"DP Simplified Taxi {taxi_id} ({len(dp_route)} pts)"
            ).add_to(group_dp_master)

            # Markers on retained DP vertices
            for pt in dp_route:
                folium.CircleMarker(
                    location=pt,
                    radius=3,
                    color=traj_color,
                    fill=True,
                    fill_color='white',
                    fill_opacity=1.0,
                    tooltip=f"Retained Point: Taxi {taxi_id}"
                ).add_to(group_dp_master)

            # End Flag Marker
            folium.Marker(
                location=dp_route[-1],
                icon=folium.Icon(color=traj_color, icon='flag'),
                tooltip=f"End: Taxi {taxi_id}"
            ).add_to(group_dp_master)

    group_orig_master.add_to(m)
    group_dp_master.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    filename = f"{file}.html"
    m.save(filename)
    print(f"✅ Interactive comparison map saved successfully: '{filename}'\n")

if __name__ == "__main__":
    file = 'tdrive_processed.csv'
    
    print("Loading data...")
    df_temp = pd.read_csv(file, header=0, parse_dates=['time'])
    
    # Filter 1-hour time window
    start_time = pd.to_datetime("2008-02-02 13:00:00")
    end_time = pd.to_datetime("2008-02-02 14:00:00")
    df_temp = df_temp[(df_temp['time'] >= start_time) & (df_temp['time'] <= end_time)].copy()

    # Grab first 50 taxis
    unique_ids = df_temp['id'].unique()[:50]
    df_50_taxi = df_temp[df_temp['id'].isin(unique_ids)].copy()

    # Apply DP Algorithm
    df_simplified = simplify_trajectories_dp(df_50_taxi, tolerance_meters=31.5)  # 31.5 meters tolerance

    print(f"Original total points: {len(df_50_taxi)}")
    print(f"Simplified total points: {len(df_simplified)}")
    print(f"Point reduction: {((len(df_50_taxi) - len(df_simplified)) / len(df_50_taxi)) * 100:.1f}%\n")

    # Visualize Both Sets
    visualize_comparison_traj(df_50_taxi, df_simplified, file="traj_map_comparison_1hr_31.5m")