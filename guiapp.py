import numpy as np
import pyvista as pv
import random
from scipy.spatial import distance
from PyQt5 import QtWidgets, QtCore
import time
import threading

class DisasterApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_simulation = None
        self.plotter = None
        self.trees = []
        self.buildings = []
        self.tree_states = []
        self.simulation_stats = {'fire_started_count': 0, 'trees_burned': 0, 'smoke_level': 0}

    def init_ui(self):
        self.setWindowTitle("Disaster Simulation")
        self.setGeometry(100, 100, 500, 400)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Dropdown menu for disaster selection
        self.label = QtWidgets.QLabel("Choose a disaster to simulate:", self)
        layout.addWidget(self.label)
        
        self.disaster_choice = QtWidgets.QComboBox(self)
        self.disaster_choice.addItems(["Earthquake", "Wildfire"])
        layout.addWidget(self.disaster_choice)
        
        # Button to load simulation scene
        self.load_scene_button = QtWidgets.QPushButton("Load Simulation Scene", self)
        self.load_scene_button.clicked.connect(self.load_simulation_scene)
        layout.addWidget(self.load_scene_button)

        # Button to trigger disaster (initially disabled)
        self.trigger_disaster_button = QtWidgets.QPushButton("Trigger Disaster", self)
        self.trigger_disaster_button.setEnabled(False)
        self.trigger_disaster_button.clicked.connect(self.run_current_disaster)
        layout.addWidget(self.trigger_disaster_button)

        # Reset button
        self.reset_button = QtWidgets.QPushButton("Reset Simulation", self)
        self.reset_button.clicked.connect(self.reset_simulation)
        layout.addWidget(self.reset_button)

        # Simulation status label
        self.status_label = QtWidgets.QLabel("", self)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

    def reset_simulation(self):
        # Reset all simulation-related attributes
        self.current_simulation = None
        self.trees = []
        self.buildings = []
        self.tree_states = []
        self.simulation_stats = {
            'fire_started_count': 0, 
            'trees_burned': 0, 
            'smoke_level': 0
        }
        
        # Close and reset plotter if it exists
        if self.plotter:
            try:
                self.plotter.close()
                self.plotter = None
            except:
                pass
        
        # Disable trigger disaster button
        self.trigger_disaster_button.setEnabled(False)
        
        # Clear status label
        self.status_label.setText("")

    def load_simulation_scene(self):
        # First reset any existing simulation
        self.reset_simulation()
        
        choice = self.disaster_choice.currentText()
        
        # Create a new plotter
        self.plotter = pv.Plotter(off_screen=False)
        self.current_simulation = choice
        
        if choice == "Earthquake":
            self.load_earthquake_scene()
        elif choice == "Wildfire":
            self.load_wildfire_scene()
        
        # Enable trigger disaster button
        self.trigger_disaster_button.setEnabled(True)
        
        # Show the plotter
        self.plotter.show()

    def load_earthquake_scene(self):
        self.plotter.add_title("Earthquake Simulation Scene")
        
        ground_surface = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=50, j_size=50)
        self.plotter.add_mesh(ground_surface, color="green", name="Ground")
        
        self.buildings = []
        
        for x in range(-20, 21, 10):
            for y in range(-20, 21, 10):
                height = random.uniform(10, 20)
                building = pv.Box(bounds=(x, x + 5, y, y + 5, 0, height))
                self.plotter.add_mesh(building, color="gray", name=f"Building_{x}_{y}")
                self.buildings.append((building, (x + 2.5, y + 2.5), height))

    def load_wildfire_scene(self):
        self.plotter.add_title("Wildfire Simulation Scene")
        
        ground_surface = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=50, j_size=50)
        self.plotter.add_mesh(ground_surface, color="green", name="Ground")
        
        green_variations = [np.array([0.3, 0.8, 0.3]), np.array([0.2, 0.6, 0.2])]
        
        self.trees = []
        self.tree_states = []
        
        for _ in range(50):
            x, y = random.uniform(-20, 20), random.uniform(-20, 20)
            tree_height = random.uniform(4.5, 6) 
            trunk = pv.Cylinder(center=(x,y,(tree_height/2)), direction=(0 ,0 ,1), radius=0.3,
                                height=tree_height) 
            foliage_layers = []
            for j in range(3): 
                foliage_height = tree_height - j * (tree_height /3) 
                foliage_layer = pv.Cone(center=(x,y,(tree_height + j * foliage_height/3)), 
                                        direction=(0 ,0 ,1), height=foliage_height - j*0.5,
                                        radius=2 - j*0.5) 
                foliage_layers.append(foliage_layer) 
            
            tree_mesh = trunk 
            for layer in foliage_layers: 
                tree_mesh += layer 
            
            self.trees.append(tree_mesh) 
            color = green_variations[random.randint(0,len(green_variations)-1)]
            self.plotter.add_mesh(tree_mesh, color=color, name=f"Tree_{x}_{y}")

            # Track tree states
            self.tree_states.append({
                'burned': False,
                'location': (x,y),
                'mesh': tree_mesh,
                'original_color': color
            })

    def run_current_disaster(self):
        if self.current_simulation == "Earthquake":
            self.simulate_earthquake()
        elif self.current_simulation == "Wildfire":
            self.simulate_wildfire()

    def simulate_earthquake(self):
        epicenter = (0, 0)
        magnitude = random.uniform(6, 8)
        depth = random.uniform(5, 15)

        for building, position, height in self.buildings:
            distance_to_epicenter = distance.euclidean(position, epicenter)
            intensity = magnitude / (1 + distance_to_epicenter / depth)

            tilt_angle = np.radians(intensity * random.uniform(0, 5))
            rotation_matrix = pv.transformations.axis_angle_rotation([1, 0, 0], tilt_angle)
            building.transform(rotation_matrix)

            shake_intensity = intensity * 0.1
            shake_noise = np.random.normal(0, shake_intensity, building.points.shape)
            building.points += shake_noise

            if intensity > 6:
                collapse_factor = random.uniform(0.1, 0.9)
                new_height = height * (1 - collapse_factor)
                building.points[:, 2] *= (new_height / height)

        self.plotter.render()
        self.status_label.setText(f"Earthquake Magnitude: {magnitude:.2f}")

    def simulate_wildfire(self):
        # Find a random fire center
        fire_center_x = random.uniform(-20, 20) 
        fire_center_y = random.uniform(-20, 20) 

        # Simulate fire spreading
        trees_burned_this_round = 0
        for tree_state in self.tree_states:
            if tree_state['burned']:
                continue

            tx, ty = tree_state['location'] 
            distance_to_fire_center = distance.euclidean((tx, ty), (fire_center_x, fire_center_y)) 

            if distance_to_fire_center < random.uniform(2, 4): 
                # Start burning the tree
                tree_state['burned'] = True
                burning_color = np.array([1.0, 0.3, 0.0])  
                
                # Update tree mesh to show burning
                tree_mesh = tree_state['mesh']
                tree_mesh.points[:,2] += random.uniform(1, 2)  
                
                # Update the mesh color
                idx = self.trees.index(tree_mesh)
                self.plotter.add_mesh(tree_mesh, color=burning_color, name=f"BurnedTree_{tx}_{ty}")
                
                # Simulate smoke
                smoke_radius = random.uniform(1, 3)  
                smoke = pv.Sphere(center=(tx, ty, 10), radius=smoke_radius)  
                self.plotter.add_mesh(smoke, color=(1.,1.,1.), opacity=0.5)
                
                trees_burned_this_round += 1

        # Update simulation stats and status
        self.simulation_stats['fire_started_count'] += 1
        self.simulation_stats['trees_burned'] += trees_burned_this_round
        self.status_label.setText(f"Trees Burned: {self.simulation_stats['trees_burned']}")
        
        # Render the changes
        self.plotter.render()

def main():
    app = QtWidgets.QApplication([])
    disaster_app = DisasterApp()
    disaster_app.show()
    app.exec_()

if __name__ == "__main__":
    main()
