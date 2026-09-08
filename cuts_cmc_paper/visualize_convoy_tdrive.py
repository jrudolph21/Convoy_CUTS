import folium
import pandas as pd

def visualize_convoys(df, convoys, start_time, end_time, target_index=None, file="convoy_map"):
    """
    Generates an interactive HTML map of Beijing showing discovered convoys with temporal overlays.
    
    Parameters:
    - df: The main dataframe of taxi trajectories.
    - convoys: The list of discovered convoy dictionaries.
    - start_time: Global analysis window start timestamp.
    - end_time: Global analysis window end timestamp.
    - target_index (int, optional): If provided, visualizes ONLY the convoy at 
      this index in the list. If None, visualizes ALL convoys.
    """
    if not convoys:
        print("No convoys to visualize.")
        return

    # 1. Filter to a specific convoy if requested
    if target_index is not None:
        if target_index < 0 or target_index >= len(convoys):
            print(f"Error: target_index {target_index} is out of bounds (0 to {len(convoys)-1}).")
            return
        
        convoys_to_plot = [(target_index, convoys[target_index])]
        filename = f"convoy_{target_index + 1}_map_beijing.html"
        print(f"Generating map for Convoy {target_index + 1}...")
    else:
        convoys_to_plot = list(enumerate(convoys))
        filename = f"{file}.html"
        print("Generating map for ALL convoys...")

    # 2. Calculate map center based ONLY on the vehicles being plotted
    active_objects = set()
    for _, c in convoys_to_plot:
        active_objects.update(c['objects'])
        
    center_lat = df[df['id'].isin(active_objects)]['y'].mean()
    center_lon = df[df['id'].isin(active_objects)]['x'].mean()

    if pd.isna(center_lat) or pd.isna(center_lon):
        center_lat, center_lon = 39.9, 116.4 
        
    # 3. Initialize Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
    
    # 4. Add floating HTML Title Card with Overall Window Start & End Times
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

    # Distinct colors for each convoy group
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'magenta', 'black']
    
    # 5. Draw the selected convoy(s)
    for original_idx, convoy in convoys_to_plot:
        convoy_objects = list(convoy['objects'])
        t_start = pd.to_datetime(convoy['start_time'])
        t_end = pd.to_datetime(convoy['end_time'])
        
        # Format duration string
        duration_mins = int((t_end - t_start).total_seconds() / 60)
        time_label = f"{t_start.strftime('%H:%M')} - {t_end.strftime('%H:%M')} ({duration_mins}m)"
        
        # Create a togglable layer group for this specific convoy
        convoy_group = folium.FeatureGroup(
            name=f"Convoy {original_idx + 1} (Size: {len(convoy_objects)} | Time: {time_label})"
        )
        
        # Filter and sort chronological data
        convoy_df = df[(df['id'].isin(convoy_objects)) & 
                       (df['time'] >= t_start) & 
                       (df['time'] <= t_end)].copy()
        convoy_df = convoy_df.sort_values(by=['id', 'time'])
        
        convoy_color = colors[original_idx % len(colors)]
        
        for obj_id in convoy_objects:
            obj_data = convoy_df[convoy_df['id'] == obj_id]
            route = list(zip(obj_data['y'], obj_data['x']))
            
            if route:
                start_pt_time = pd.to_datetime(obj_data['time'].iloc[0]).strftime('%H:%M:%S')
                end_pt_time = pd.to_datetime(obj_data['time'].iloc[-1]).strftime('%H:%M:%S')

                # 1. Trajectory line with exact timestamps in tooltip
                folium.PolyLine(
                    route,
                    weight=5,
                    color=convoy_color,
                    opacity=0.75,
                    tooltip=f"Convoy {original_idx + 1} | Taxi {obj_id} ({start_pt_time} - {end_pt_time})"
                ).add_to(convoy_group)
                
                # 2. Start marker displaying start timestamp
                folium.CircleMarker(
                    location=route[0],
                    radius=5,
                    color=convoy_color,
                    fill=True,
                    fill_color='white',
                    fill_opacity=1.0,
                    tooltip=f"Start: Taxi {obj_id} at {start_pt_time}"
                ).add_to(convoy_group)
                
                # 3. End marker displaying end timestamp
                folium.Marker(
                    location=route[-1],
                    icon=folium.Icon(color=convoy_color, icon='flag'),
                    tooltip=f"End: Taxi {obj_id} at {end_pt_time}"
                ).add_to(convoy_group)
        
        convoy_group.add_to(m)
        
    # Layer control panel (Top Right)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save the file
    m.save(filename)
    print(f"✅ Interactive map saved successfully: '{filename}'\n")