import pyvista as pv
import numpy as np
import random
import requests
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import networkx as nx


API_KEY = 'add19785e49c43dd4ebef4df0930a16a'
CITY = 'Pune'

class DisasterPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.historical_data = []
    def generate_synthetic_data(self, num_samples=1000):
        np.random.seed(42)
        data = {
        'magnitude': np.random.uniform(4.0, 9.0, num_samples),
        'depth': np.random.uniform(5.0, 50.0, num_samples),
        'building_density': np.random.uniform(0.3, 0.9, num_samples),
        'soil_condition': np.random.uniform(0.1, 1.0, num_samples),
        'weather_temp': np.random.uniform(10, 40, num_samples),
        'weather_humidity': np.random.uniform(30, 90, num_samples)
        }
        damage = (
        data['magnitude'] * 1.5 +
        -0.3 * data['depth'] +
        0.5 * data['building_density'] +
        0.7 * data['soil_condition'] +
        0.1 * data['weather_temp'] / 40 +
        0.1 * data['weather_humidity'] / 100
        )
        data['damage_level'] = damage
        return pd.DataFrame(data)
    def train_model(self):
        df = self.generate_synthetic_data()
        X = df.drop('damage_level', axis=1)
        y = df['damage_level']
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        joblib.dump(self.model, 'disaster_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
    def predict_damage(self, features):
        feature_names = ['magnitude', 'depth', 'building_density', 'soil_condition', 'weather_temp', 'weather_humidity']
        features_df = pd.DataFrame([features], columns=feature_names)
        features_scaled = self.scaler.transform(features_df)
        prediction = self.model.predict(features_scaled)
        return prediction[0]
    def update_historical_data(self, earthquake_data, actual_damage):
        self.historical_data.append({
        'timestamp': datetime.now(),
        'magnitude': earthquake_data['magnitude'],
        'depth': earthquake_data['depth'],
        'predicted_damage': earthquake_data['predicted_damage'],
        'actual_damage': actual_damage
        })
        if len(self.historical_data) >= 50:
            self.train_model()
            self.historical_data = []

def create_smart_city_simulation():
    disaster_predictor = DisasterPredictor()
    disaster_predictor.train_model()
    plotter = pv.Plotter()


    def identify_optimal_safe_zone(buildings, water):
        # Exclude areas near water and buildings
        safe_candidates = []
        grid_resolution = 50
        x_range = np.linspace(-20, 20, grid_resolution)
        y_range = np.linspace(-20, 20, grid_resolution)


        for x in x_range:
            for y in y_range:
                # Check distance from buildings and water
                building_distances = [distance.euclidean([x, y], [b[1][0], b[1][1]]) for b in buildings]
                water_distance = distance.euclidean([x, y], [water.center[0], water.center[1]])
               
                if (min(building_distances) > 15 and water_distance > 25):
                    safe_candidates.append([x, y])


        # Choose center of the safest region
        if safe_candidates:
            return np.mean(safe_candidates, axis=0)
        return [15, 15]  # Fallback to previous location


    def create_safe_route_network(buildings, safe_zone_center):
    # Create a flexible graph with precise node placement
        G = nx.Graph()
   
    # Create nodes with fine-grained control
        node_spacing = 1  # Smaller spacing for more precise routing
        x_range = range(-20, 21, node_spacing)
        y_range = range(-20, 21, node_spacing)
    
        # Add nodes to the graph
        for x in x_range:
            for y in y_range:
                G.add_node((x, y))
    
        # Connect adjacent nodes
        for x in x_range:
            for y in y_range:
                # Connect horizontally and vertically
                if x+node_spacing in x_range:
                    G.add_edge((x,y), (x+node_spacing,y))
                if y+node_spacing in y_range:
                    G.add_edge((x,y), (x,y+node_spacing))
    
        # Remove nodes near buildings with precise obstacle avoidance
        for building, building_center, _ in buildings:
            bx, by = int(building_center[0]), int(building_center[1])
            for x in x_range:
                for y in y_range:
                    # Calculate precise distance to building
                    distance_to_building = np.sqrt((x-bx)**2 + (y-by)**2)
                    if distance_to_building < 3 and (x,y) in G.nodes():
                        G.remove_node((x,y))
    
        # Ensure safe zone and start points are in the graph
        safe_node = (int(safe_zone_center[0]), int(safe_zone_center[1]))
        start_points = [(-15, -15), (0, -15), (15, -15),
                        (-15, 0), (15, 0),
                        (-15, 15), (0, 15), (15, 15)]
    
        # Intelligent node connection strategy
        for point in start_points + [safe_node]:
            if point not in G.nodes():
                # Find nearest existing nodes and connect them
                nearest_nodes = sorted(
                    G.nodes(), 
                    key=lambda n: np.linalg.norm(np.array(n) - np.array(point))
                )
                
                # Connect to the 3 nearest nodes
                for nearest_node in nearest_nodes[:3]:
                    # Add the point as a new node
                    G.add_node(point)
                    # Connect to the nearest nodes
                    G.add_edge(point, nearest_node)
                    
                    # Attempt to find a path to safe zone
                    try:
                        nx.shortest_path(G, point, safe_node)
                        break  # If path found, stop adding connections
                    except nx.NetworkXNoPath:
                        continue
    
        return G, safe_node
    # Create a larger grid for better road placement
    grid_size = 10
    grid_spacing = 40 / grid_size

    
    def visualize_safe_routes(route_graph, safe_zone_center, plotter):
        start_points = [
        (-15, -15), (0, -15), (15, -15),   # Bottom row
        (-15, 0), (15, 0),                 # Middle horizontal
        (-15, 15), (0, 15), (15, 15)       # Top row
    ]
    # Target safe zone
        target = (int(safe_zone_center[0]), int(safe_zone_center[1]))
   
    # Collect all route lines
        route_lines = []
   
    # Generate routes for all start points
        for start in start_points:
            try:
            # Find the shortest path
                path = nx.shortest_path(route_graph, start, target)
           
            # Create line segments for the path
                for i in range(len(path) - 1):
                    line = pv.Line(
                        (path[i][0], path[i][1], 0.5),  # Slightly raised for visibility
                        (path[i+1][0], path[i+1][1], 0.5)
                    )
                    route_lines.append(line)
                
            except nx.NetworkXNoPath:
                    print(f"No path found from {start} to {target}")
   
    # Combine and visualize routes
        if route_lines:
        # Try to remove the previous routes first
            try:
                plotter.remove_actor('safe_routes')
            except:
                pass
        
            combined_routes = pv.MultiBlock(route_lines)
            plotter.add_mesh(
                combined_routes,
                color='lime',
                line_width=10,  # Thick, visible lines
                opacity=0.8,    # High visibility
                name='safe_routes'  # Use a consistent name for removal
            )
    # Define start points
    def create_water_body():
        water = pv.Plane(center=(0, 0, -0.5), i_size=200, j_size=200)
        return water
    water = create_water_body()
    plotter.add_mesh(water, color='darkblue', opacity=0.6)

    def create_tectonic_plates():
        plate1 = pv.Plane(center=(0, -25, -1), i_size=100, j_size=50)
        plate2 = pv.Plane(center=(0, 25, -1), i_size=100, j_size=50)
        return plate1, plate2

    plate1, plate2 = create_tectonic_plates()
    #plotter.add_mesh(plate1, color='brown', opacity=0.5)
    #plotter.add_mesh(plate2, color='brown', opacity=0.5)

    # Create a larger ground plane
    ground = pv.Plane(i_size=100, j_size=100)
    ground_surface = ground.elevation(low_point=(0, 0, 0), high_point=(0, 0, 1))
    plotter.add_mesh(ground_surface, color='green', show_edges=True)

    # Create a larger grid for better road placement
    grid_size = 10
    grid_spacing = 40 / grid_size

    def get_random_building_height():
        return random.uniform(2.0, 6.0)

    def get_random_color():
        return [random.random() for _ in range(3)]

    def create_human_figure(x, y, z):
        body = pv.Cylinder(center=(x, y, z + 0.75), direction=(0, 0, 1), height=1.5, radius=0.25)
        head = pv.Sphere(center=(x, y, z + 1.75), radius=0.25)
        leg_left = pv.Cylinder(center=(x - 0.15, y, z + 0.4), direction=(0, 0, 1), height=0.8, radius=0.1)
        leg_right = pv.Cylinder(center=(x + 0.15, y, z + 0.4), direction=(0, 0, 1), height=0.8, radius=0.1)
        return body + head + leg_left + leg_right

    def create_road_network(grid_size, grid_spacing):
        roads = []
        for i in range(grid_size + 1):
            x = -20 + i * grid_spacing
            road = ((x, -20, 0.1), (x, 20, 0.1))
            roads.append(road)
        for j in range(grid_size + 1):
            y = -20 + j * grid_spacing
            road = ((-20, y, 0.1), (20, y, 0.1))
            roads.append(road)
        return roads
    # Create road network
    roads = create_road_network(grid_size, grid_spacing)
    for road in roads:
        line = pv.Line(road[0], road[1])
        plotter.add_mesh(line, color='gray', line_width=5)

    buildings = []
    building_density = 0.0
    for i in range(grid_size):
        for j in range(grid_size):
            if random.random() > 0.7:  # 30% chance of placing a building
                building_density += 1
                x = -20 + i * grid_spacing + random.uniform(-1, 1)
                y = -20 + j * grid_spacing + random.uniform(-1, 1)
                height = get_random_building_height()
                building = pv.Cube(center=[x, y, height / 2], x_length=2, y_length=2, z_length=height)
                buildings.append((building, [x, y, height / 2], height))
                plotter.add_mesh(building, color=get_random_color())

    safe_zone_center = [15, 15, 0.1]  # Safe zone coordinates within the city grid
    safe_zone = pv.Cube(center=safe_zone_center, x_length=8, y_length=8, z_length=0.2)
    plotter.add_mesh(safe_zone, color='green', opacity=0.7)  # Safe zone is green

    # Create a circular boundary
    # Create a series of points in a circle
    theta = np.linspace(0, 2*np.pi, 20)  # Reduced number of points for simplicity
    radius = 5
    lines = []

    for i in range(len(theta) - 1):
        start_point = [
            safe_zone_center[0] + radius * np.cos(theta[i]), 
            safe_zone_center[1] + radius * np.sin(theta[i]), 
            safe_zone_center[2]
        ]
        end_point = [
            safe_zone_center[0] + radius * np.cos(theta[i+1]), 
            safe_zone_center[1] + radius * np.sin(theta[i+1]), 
            safe_zone_center[2]
        ]
        
        line = pv.Line(start_point, end_point)
        lines.append(line)

    # Combine lines into a multiblock dataset
    safe_zone_boundary = pv.MultiBlock(lines)

    # Add the boundary to the plotter
    plotter.add_mesh(safe_zone_boundary, color='white', opacity=1, line_width=5)
    humans = []
    for _ in range(20):
        x, y = np.random.uniform(-20, 20, size=2)
        human = create_human_figure(x, y, 0)
        humans.append((human, [x, y, 0]))
        plotter.add_mesh(human, color=get_random_color())

    rescue_vehicles = []
    for _ in range(3):  # Create 3 rescue vehicles
        x, y = np.random.uniform(-20, 20, size=2)
        vehicle = pv.Cube(center=[x, y, 0.5], x_length=3, y_length=2, z_length=1)
        rescue_vehicles.append(vehicle)
        plotter.add_mesh(vehicle, color='red')

    def move_humans_via_safe_routes(humans, route_graph, safe_zone_center):
        # Precise safe zone coordinates
        safe_zone_node = (int(safe_zone_center[0]), int(safe_zone_center[1]))
       
        # Track humans that have reached the safe zone
        moved_humans = []
       
        for human, position in humans:
            # Convert current position to graph node
            current_node = (int(round(position[0])), int(round(position[1])))
           
            # Ensure the current node exists in the graph
            if current_node not in route_graph.nodes():
                current_node = min(route_graph.nodes(), key=lambda n: np.linalg.norm(np.array(n) - np.array(current_node)))
           
            try:
                # Find the shortest path to the safe zone
                path = nx.shortest_path(route_graph, current_node, safe_zone_node)
               
                # If we're not at the safe zone, move along the path
                if len(path) > 1:
                    next_node = path[1]
                   
                    # Calculate movement vector
                    movement = np.array([next_node[0] - position[0], next_node[1] - position[1], 0]) * 1.0
                   
                    # Move the human
                    human.translate(movement, inplace=True)
                   
                    # Update position
                    new_position = [next_node[0], next_node[1], position[2]]
                    moved_humans.append((human, new_position))
                else:
                    # Already at the safe zone
                    moved_humans.append((human, position))
           
            except nx.NetworkXNoPath:
                # Fallback: direct movement to safe zone if no path found
                direction = np.array([safe_zone_center[0] - position[0],
                                    safe_zone_center[1] - position[1], 0])
               
                movement_magnitude = np.linalg.norm(direction)
               
                if movement_magnitude > 1:
                    # Normalize and scale movement
                    movement = direction / movement_magnitude * 1.0
                    human.translate(movement, inplace=True)
                    new_position = position + movement
                    moved_humans.append((human, new_position))
                else:
                    # Very close to safe zone
                    moved_humans.append((human, position))
       
        return moved_humans


    def show_seismic_waves(epicenter, magnitude):
        wave_radius = magnitude * 5  # Simple correlation between magnitude and wave radius
        num_waves = int(magnitude)
        for i in range(num_waves):
            radius = i * wave_radius / num_waves
            wave = pv.Cylinder(center=(epicenter[0], epicenter[1], 0), direction=(0, 0, 1), height=0.1, radius=radius)
            plotter.add_mesh(wave, color='orange', opacity=0.1 + 0.1 * (num_waves - i))

    def simulate_earthquake():
        nonlocal blinking_boundary
        magnitude = random.uniform(4.0, 9.0)
        epicenter = np.array([random.uniform(-20, 20), random.uniform(-20, 20), 0])
        depth = random.uniform(5.0, 50.0)
        blinking_boundary.on_earthquake()
    
        def get_weather_data():
            url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
            response = requests.get(url)
            data = response.json()
            if response.status_code == 200:
                weather_data = {
                    'Temperature': data['main']['temp'],
                    'Humidity': data['main']['humidity'],
                    'Pressure': data['main']['pressure'],
                    'Wind Speed': data['wind']['speed'],
                    'Weather': data['weather'][0]['description'],
                }
                return weather_data
            else:
                print("Error fetching weather data:", data.get('message', 'Unknown error'))
                return {}

        # Function to display weather data
        def display_weather_data(data):
            text = (f"Temperature: {data['Temperature']} °C\n"
                    f"Humidity: {data['Humidity']}%\n"
                    f"Pressure: {data['Pressure']} hPa\n"
                    f"Wind Speed: {data['Wind Speed']} m/s\n"
                    f"Weather: {data['Weather']}")
            plotter.add_text(text, position='upper_left', font_size=10, color='white', name='weather_info')
        weather_data = get_weather_data()

        features = [
            magnitude,
            depth,
            building_density,
            0.7,  # soil condition (assumed constant for this example)
            weather_data.get('Temperature', 25),  # default if weather data unavailable
            weather_data.get('Humidity', 60)  # default if weather data unavailable
        ]
        # Get ML prediction for damage
        predicted_damage = disaster_predictor.predict_damage(features)
        # Move tectonic plates
        plate1.translate((0, -1, 0), inplace=True)
        plate2.translate((0, 1, 0), inplace=True)
        # Crumble ground
        ground_points = ground_surface.points
        noise = np.random.normal(0, magnitude * 0.05, ground_points.shape)
        ground_surface.points = ground_points + noise

        # Track actual damage
        total_damage = 0
        buildings_affected = 0

        for building, position, height in buildings:
            distance_to_epicenter = distance.euclidean(position, epicenter)
            intensity = magnitude / (1 + distance_to_epicenter / depth)

            # Apply tilting effect
            tilt_angle = np.radians(intensity * random.uniform(0, 5))
            rotation_matrix = pv.transformations.axis_angle_rotation([1, 0, 0], tilt_angle)
            building.transform(rotation_matrix, inplace=True)

            # Apply shaking effect
            shake_intensity = intensity * 0.1
            shake_noise = np.random.normal(0, shake_intensity, building.points.shape)
            building.points += shake_noise

            # Calculate damage for this building
            if intensity > 6:
                collapse_factor = random.uniform(0.1, 0.9)
                total_damage += collapse_factor
                buildings_affected += 1
                new_height = height * (1 - collapse_factor)
                building.points[:, 2] *= (new_height / height)
                building.points[:, 2] += 0.1  # Prevent buildings from sinking into the ground
        # Calculate actual damage level
        actual_damage = (total_damage / len(buildings)) * 10 if buildings else 0

        # Update ML model with actual damage data
        disaster_predictor.update_historical_data({
            'magnitude': magnitude,
            'depth': depth,
            'predicted_damage': predicted_damage
        }, actual_damage)

        for human, position in humans:
            distance_to_epicenter = distance.euclidean(position, epicenter)
            intensity = magnitude / (1 + distance_to_epicenter / depth)
            # Apply movement effect
            movement = np.random.normal(0, intensity * 0.1, 3)
            human.translate(movement, inplace=True)

        # Simulate earthquake data display
        earthquake_data = {
            'Magnitude': magnitude,
            'Epicenter': f"({epicenter[0]:.2f}, {epicenter[1]:.2f})",
            'Depth': depth,
        }
        earthquake_text = (f"Earthquake Data:\n"
                           f"Magnitude: {earthquake_data['Magnitude']:.2f}\n"
                           f"Epicenter: {earthquake_data['Epicenter']}\n"
                           f"Depth: {earthquake_data['Depth']:.2f} km")
        plotter.add_text(earthquake_text, position='lower_left', font_size=10, color='red', name='earthquake_info')

        # ML Predictions and Analytics
        analytics_text = (f"ML Predictions & Analytics:\n"
                          f"Predicted Damage: {predicted_damage:.2f}\n"
                          f"Actual Damage: {actual_damage:.2f}\n"
                          f"Buildings Affected: {buildings_affected}\n"
                          f"Prediction Accuracy: {100 - abs(predicted_damage - actual_damage) * 10:.1f}%")
        plotter.add_text(analytics_text, position='lower_right', font_size=10, color='yellow', name='analytics_info')

        # Calculate risk zones based on ML predictions
        risk_level = 'High' if predicted_damage > 7 else 'Medium' if predicted_damage > 5 else 'Low'
        estimated_recovery_time = int(predicted_damage * 30)  # days

        # Generate evacuation routes based on damage predictions
        priority_zones = []
        if predicted_damage > 7:
            priority_zones = ['A1', 'A2', 'B1']  # High-risk areas
        elif predicted_damage > 5:
            priority_zones = ['B1', 'B2']  # Medium-risk areas
        else:
            priority_zones = ['C1']  # Low-risk areas

        # Calculate resource requirements using ML predictions
        required_resources = {
            'emergency_vehicles': int(predicted_damage * 10),
            'medical_teams': int(predicted_damage * 5),
            'rescue_workers': int(predicted_damage * 20),
            'shelter_capacity': int(predicted_damage * 100)
        }
        # Disaster management recommendations based on ML insights
        management_text = (f"Disaster Management Plan:\n"
                           f"Risk Level: {risk_level}\n"
                           f"Priority Zones: {', '.join(priority_zones)}\n"
                           f"Required Resources:\n"
                           f"- Emergency Vehicles: {required_resources['emergency_vehicles']}\n"
                           f"- Medical Teams: {required_resources['medical_teams']}\n"
                           f"- Rescue Workers: {required_resources['rescue_workers']}\n"
                           f"- Shelter Capacity: {required_resources['shelter_capacity']}\n"
                           f"Est. Recovery: {estimated_recovery_time} days")
        plotter.add_text(management_text, position='upper_right', font_size=10, color='cyan', name='management_info')

        # Move the rescue vehicles toward the epicenter
        for vehicle in rescue_vehicles:
            target_x, target_y = epicenter[0], epicenter[1]
            direction = np.array([target_x - vehicle.center[0], target_y - vehicle.center[1], 0])
            movement = direction / np.linalg.norm(direction) * 0.3
            vehicle.translate(movement, inplace=True)
        # Show seismic waves and earthquake shaking
        visualize_safe_routes(route_graph,safe_zone_center, plotter)
        show_seismic_waves(epicenter, magnitude)
        plotter.render()

    route_graph, safe_zone = create_safe_route_network(buildings, safe_zone_center)

    class BlinkingSafeZoneBoundary:
        def __init__(self, plotter, safe_zone_center):
            self.plotter = plotter
            self.safe_zone_center = safe_zone_center
            self.boundary = None
            self.is_visible = False
            self.blink_interval = 10  # Number of render calls between blinks
            self.blink_counter = 0
            self.people_in_safe_zone = 0

        def create_blinking_boundary(self):
            # Create a circular boundary
            theta = np.linspace(0, 2*np.pi, 20)
            radius = 5
            lines = []

            for i in range(len(theta) - 1):
                start_point = [
                    self.safe_zone_center[0] + radius * np.cos(theta[i]), 
                    self.safe_zone_center[1] + radius * np.sin(theta[i]), 
                    self.safe_zone_center[2]
                ]
                end_point = [
                    self.safe_zone_center[0] + radius * np.cos(theta[i+1]), 
                    self.safe_zone_center[1] + radius * np.sin(theta[i+1]), 
                    self.safe_zone_center[2]
                ]
                
                line = pv.Line(start_point, end_point)
                lines.append(line)

            # Combine lines into a multiblock dataset
            self.boundary = pv.MultiBlock(lines)

        def update_boundary(self, humans):
            # Create boundary if not exists
            if self.boundary is None:
                self.create_blinking_boundary()

            # Count people in safe zone
            self.people_in_safe_zone = sum(1 for human, position in humans 
                                           if (np.abs(position[0] - self.safe_zone_center[0]) < 5 and 
                                               np.abs(position[1] - self.safe_zone_center[1]) < 5))

            # Trigger blinking when most people are in safe zone
            if self.people_in_safe_zone > len(humans) * 0.7:
                self.blink_counter += 1

                # Blink every blink_interval calls
                if self.blink_counter % self.blink_interval < self.blink_interval / 2:
                    if not self.is_visible:
                        self.plotter.add_mesh(
                            self.boundary, 
                            color='white', 
                            opacity=1, 
                            line_width=5,
                            name='blinking_safe_zone_boundary'
                        )
                        self.is_visible = True
                else:
                    try:
                        self.plotter.remove_actor('blinking_safe_zone_boundary')
                        self.is_visible = False
                    except:
                        pass

        def on_earthquake(self):
            # Create boundary if not exists
            if self.boundary is None:
                self.create_blinking_boundary()

            # Ensure boundary is always visible during earthquake
            try:
                self.plotter.remove_actor('blinking_safe_zone_boundary')
            except:
                pass
            
            # Add boundary with blinking effect
            self.plotter.add_mesh(
                self.boundary, 
                color='red', 
                opacity=1, 
                line_width=7,
                name='blinking_safe_zone_boundary'
            )

    blinking_boundary = BlinkingSafeZoneBoundary(plotter, safe_zone_center)

    # Continuously move humans, even during the earthquake
    def update_simulation():
        nonlocal humans, blinking_boundary
        humans = move_humans_via_safe_routes(humans, route_graph, safe_zone_center)
        blinking_boundary.update_boundary(humans)
        plotter.render()  # Update the plot

    # Add the earthquake simulation to the 'e' key event
    plotter.add_key_event('e', simulate_earthquake)
    # After creating buildings and safe zone
    route_graph, safe_zone = create_safe_route_network(buildings, safe_zone_center)
    
    plotter.show(auto_close=False)
    while True:
        update_simulation()
        plotter.update()
if __name__ == "__main__":
    create_smart_city_simulation()
