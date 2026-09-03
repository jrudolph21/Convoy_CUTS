
import folium
import pandas as pd

def visualize_convoys(df, convoys, target_index=None, file="convoy_map"):
    """
    Generates an interactive HTML map of Beijing showing discovered convoys.
    
    Parameters:
    - df: The main dataframe of taxi trajectories.
    - convoys: The list of discovered convoy dictionaries.
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
        
        # Keep the original index for labeling purposes, but only plot one
        convoys_to_plot = [(target_index, convoys[target_index])]
        filename = f"convoy_{target_index + 1}_map_beijing.html"
        print(f"Generating map for Convoy {target_index + 1}...")
    else:
        # Plot everything
        convoys_to_plot = list(enumerate(convoys))
        filename = f"{file}.html"
        print("Generating map for ALL convoys...")

    # 2. Calculate map center based ONLY on the vehicles being plotted
    active_objects = set()
    for _, c in convoys_to_plot:
        active_objects.update(c['objects'])
        
    center_lat = df[df['id'].isin(active_objects)]['y'].mean()
    center_lon = df[df['id'].isin(active_objects)]['x'].mean()

    # Fallback to roughly Beijing's center if data is missing
    if pd.isna(center_lat) or pd.isna(center_lon):
        center_lat, center_lon = 39.9, 116.4 
        
    # 3. Initialize Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
    
    # Distinct colors for each convoy group
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'magenta', 'black']
    
    # 4. Draw the selected convoy(s)
    for original_idx, convoy in convoys_to_plot:
        convoy_objects = list(convoy['objects'])
        t_start = convoy['start_time']
        t_end = convoy['end_time']
        
        # Format the time for the layer label
        time_label = f"{t_start.strftime('%H:%M')} - {t_end.strftime('%H:%M')}"
        
        # Create a togglable layer group for this specific convoy
        convoy_group = folium.FeatureGroup(
            name=f"Convoy {original_idx + 1} (Size: {len(convoy_objects)} | Time: {time_label})"
        )
        
        # Filter and sort chronological data
        convoy_df = df[(df['id'].isin(convoy_objects)) & 
                       (df['time'] >= t_start) & 
                       (df['time'] <= t_end)].copy()
        convoy_df = convoy_df.sort_values(by=['id', 'time'])
        
        # Assign a single base color for all taxis in this convoy
        convoy_color = colors[original_idx % len(colors)]
        
        for obj_id in convoy_objects:
            obj_data = convoy_df[convoy_df['id'] == obj_id]
            route = list(zip(obj_data['y'], obj_data['x']))
            
            if route:
                # 1. Trajectory line
                folium.PolyLine(
                    route,
                    weight=5,
                    color=convoy_color,
                    opacity=0.75,
                    tooltip=f"Convoy {original_idx + 1} | Taxi {obj_id}"
                ).add_to(convoy_group)
                
                # 2. Start marker (white dot with colored border)
                folium.CircleMarker(
                    location=route[0],
                    radius=5,
                    color=convoy_color,
                    fill=True,
                    fill_color='white',
                    fill_opacity=1.0,
                    tooltip=f"Start: Taxi {obj_id}"
                ).add_to(convoy_group)
                
                # 3. End marker (colored flag)
                folium.Marker(
                    location=route[-1],
                    icon=folium.Icon(color=convoy_color, icon='flag'),
                    tooltip=f"End: Taxi {obj_id}"
                ).add_to(convoy_group)
        
        # Add the completed convoy layer to the map
        convoy_group.add_to(m)
        
    # Add the interactive layer control panel (Top Right)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save the file
    m.save(filename)
    print(f"✅ Interactive map saved successfully: '{filename}'\n")