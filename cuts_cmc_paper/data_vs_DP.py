import pandas as pd
import numpy as np
import time
from sklearn.cluster import DBSCAN
import folium

def visualize_traj(df, start_time, end_time,trajs=None, target_index=None, file="traj_map"):
    """
    Generates an interactive HTML map showing taxi trajectories.
    
    Parameters:
    - df (DataFrame): DataFrame containing telemetry data ('id', 'x', 'y', 'time').
    - trajs (list, optional): List of trajectory/convoy dicts. If None, generates one 
                              trajectory per unique taxi ID in `df`.
    - target_index (int, optional): If provided, visualizes ONLY the trajectory at 
                                     this index in `trajs`.
    - file (str): Base name for the saved HTML map file.
    """
    if df is None or df.empty:
        print("No data available to visualize.")
        return

    # If no convoy/trajectory list is provided, create one per unique taxi ID
    if trajs is None:
        trajs = []
        for taxi_id, group in df.groupby('id'):
            trajs.append({
                'objects': [taxi_id],
                'start_time': group['time'].min(),
                'end_time': group['time'].max()
            })

    if not trajs:
        print("No trajectories to visualize.")
        return

    # 1. Filter to a specific trajectory if requested
    if target_index is not None:
        if target_index < 0 or target_index >= len(trajs):
            print(f"Error: target_index {target_index} is out of bounds (0 to {len(trajs)-1}).")
            return
        
        trajs_to_plot = [(target_index, trajs[target_index])]
        filename = f"trajectory_{target_index + 1}_map_beijing.html"
        print(f"Generating map for Trajectory {target_index + 1}...")
    else:
        trajs_to_plot = list(enumerate(trajs))
        filename = f"{file}.html"
        print("Generating map for ALL trajectories...")

    # 2. Calculate map center based ONLY on the vehicles being plotted
    active_objects = set()
    for _, c in trajs_to_plot:
        active_objects.update(c['objects'])
        
    active_df = df[df['id'].isin(active_objects)]
    center_lat = active_df['y'].mean()
    center_lon = active_df['x'].mean()

    # Fallback to Beijing's center if coordinates are missing
    if pd.isna(center_lat) or pd.isna(center_lon):
        center_lat, center_lon = 39.9, 116.4 
        
    # 3. Initialize Folium map
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
    
    # Valid Folium icon colors
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'pink', 'black']
    
    # 4. Draw selected trajectory(ies)
    for original_idx, traj in trajs_to_plot:
        traj_objects = list(traj['objects'])
        t_start = traj['start_time']
        t_end = traj['end_time']
        
        # Format layer label
        time_label = f"{t_start.strftime('%H:%M')} - {t_end.strftime('%H:%M')}"
        
        traj_group = folium.FeatureGroup(
            name=f"Trajectory {original_idx + 1} (Size: {len(traj_objects)} | Time: {time_label})"
        )
        
        # Filter and sort data chronologically
        traj_df = df[(df['id'].isin(traj_objects)) & 
                     (df['time'] >= t_start) & 
                     (df['time'] <= t_end)].copy()
        traj_df = traj_df.sort_values(by=['id', 'time'])
        
        traj_color = colors[original_idx % len(colors)]
        
        for obj_id in traj_objects:
            obj_data = traj_df[traj_df['id'] == obj_id]
            route = list(zip(obj_data['y'], obj_data['x']))
            
            if route:
                # 1. Trajectory line
                folium.PolyLine(
                    route,
                    weight=5,
                    color=traj_color,
                    opacity=0.75,
                    tooltip=f"Trajectory {original_idx + 1} | Taxi {obj_id}"
                ).add_to(traj_group)
                
                # 2. Start marker (white dot with colored border)
                folium.CircleMarker(
                    location=route[0],
                    radius=5,
                    color=traj_color,
                    fill=True,
                    fill_color='white',
                    fill_opacity=1.0,
                    tooltip=f"Start: Taxi {obj_id}"
                ).add_to(traj_group)
                
                # 3. End marker (colored flag)
                folium.Marker(
                    location=route[-1],
                    icon=folium.Icon(color=traj_color, icon='flag'),
                    tooltip=f"End: Taxi {obj_id}"
                ).add_to(traj_group)
        
        traj_group.add_to(m)
        
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(filename)
    print(f"✅ Interactive map saved successfully: '{filename}'\n")


if __name__ == "__main__":
    file = 'tdrive_processed.csv'
    
    print("Loading data...")
    df_temp = pd.read_csv(file, header=0, parse_dates=['time'])
    
    # Filter by 5-hour time window
    start_time = pd.to_datetime("2008-02-02 13:00:00")
    end_time = pd.to_datetime("2008-02-02 18:00:00")
    
    df_temp = df_temp[(df_temp['time'] >= start_time) & (df_temp['time'] <= end_time)].copy()
    print(f"Total rows in time window: {len(df_temp)}")
    print(f"Total unique taxis in window: {df_temp['id'].nunique()}")
    
    # Grab first 20 taxis
    unique_ids = df_temp['id'].unique()[:20]
    df_20_taxi = df_temp[df_temp['id'].isin(unique_ids)].copy()
    print(df_20_taxi)
    
    # Visualize 20 taxi trajectories directly
    if not df_20_taxi.empty:
        visualize_traj(df_20_taxi, start_time, end_time, target_index=11, file="traj_taxi_paper_param")