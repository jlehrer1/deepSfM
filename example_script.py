import pycolmap
from deepsfm.deepsfm import reconstruct_images
from deepsfm.visualize3d import plot_reconstruction, init_figure
from pathlib import Path
from torchvision.transforms import Resize, Compose, PILToTensor
import plotly.io as pio
import open3d as o3d 
import os 
import numpy as np

import pycolmap
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from deepsfm.mesh_creator import CameraViewGroundDetector, create_and_preview_rl_environment
import sys
import os
import numpy as np
import open3d as o3d
import habitat_sim
import magnum as mn
from habitat_sim.utils import viz_utils as vut

import os
import subprocess

# Make sure to get the libomp path using brew
libomp_prefix = subprocess.check_output(['brew', '--prefix', 'libomp']).decode().strip()

# Set environment variables
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['DYLD_LIBRARY_PATH'] = f"{libomp_prefix}/lib:" + os.environ.get('DYLD_LIBRARY_PATH', '')

print("Environment variables set:")
print(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
print(f"KMP_DUPLICATE_LIB_OK={os.environ['KMP_DUPLICATE_LIB_OK']}")
print(f"DYLD_LIBRARY_PATH={os.environ['DYLD_LIBRARY_PATH']}")

class HabitatEnvSetup:
    def __init__(self, rl_env, output_dir="habitat_scene"):
        """
        Initialize Habitat environment setup from existing RL environment
        
        Args:
            rl_env: Output from create_and_preview_rl_environment()
            output_dir: Directory to save scene files
        """
        self.rl_env = rl_env
        self.output_dir = output_dir
        self.scene_path = None
        self.sim = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def export_scene_mesh(self, format="obj"):  # Changed default to OBJ
        """
        Export the RL environment mesh to a format Habitat-Sim can use
        
        Args:
            format: "ply" or "obj" 
        """
        print(f"=== Exporting Scene to {format.upper()} ===")
        
        # Combine all environment meshes
        combined_mesh = self.rl_env['environment_mesh'] + self.rl_env['ground_mesh']
        
        # Add walls if they exist
        if self.rl_env['walls']:
            for wall in self.rl_env['walls']:
                combined_mesh += wall
        
        # Clean up the mesh
        combined_mesh.remove_degenerate_triangles()
        combined_mesh.remove_duplicated_triangles()
        combined_mesh.remove_duplicated_vertices()
        combined_mesh.compute_vertex_normals()
        
        # Convert vertices to float32 for Habitat-Sim compatibility
        vertices = np.asarray(combined_mesh.vertices).astype(np.float32)
        triangles = np.asarray(combined_mesh.triangles).astype(np.int32)
        
        # Create new mesh with correct data types
        export_mesh = o3d.geometry.TriangleMesh()
        export_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        export_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Copy colors if available
        if combined_mesh.has_vertex_colors():
            colors = np.asarray(combined_mesh.vertex_colors).astype(np.float32)
            export_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Copy normals if available
        if combined_mesh.has_vertex_normals():
            normals = np.asarray(combined_mesh.vertex_normals).astype(np.float32)
            export_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        else:
            export_mesh.compute_vertex_normals()
        
        # Export to file
        if format == "ply":
            self.scene_path = os.path.join(self.output_dir, "scene.ply")
            success = o3d.io.write_triangle_mesh(self.scene_path, export_mesh, 
                                               write_ascii=True, write_vertex_normals=True)
        elif format == "obj":
            self.scene_path = os.path.join(self.output_dir, "scene.obj")
            success = o3d.io.write_triangle_mesh(self.scene_path, export_mesh, 
                                               write_ascii=True, write_vertex_normals=True)
        else:
            raise ValueError("Format must be 'ply' or 'obj'")
            
        if success:
            print(f"Scene exported successfully to: {self.scene_path}")
            print(f"Vertices: {len(export_mesh.vertices)} (float32)")
            print(f"Triangles: {len(export_mesh.triangles)} (int32)")
            print(f"Vertex normals: {export_mesh.has_vertex_normals()}")
            print(f"Vertex colors: {export_mesh.has_vertex_colors()}")
        else:
            raise RuntimeError(f"Failed to export scene to {format}")
            
        return self.scene_path
    
    def create_habitat_config(self, enable_physics=True, agent_radius=0.1, agent_height=1.5):
        """
        Create Habitat-Sim configuration for physics simulation
        
        Args:
            enable_physics: Enable physics simulation
            agent_radius: Agent cylinder radius for navigation
            agent_height: Agent height
        """
        print(f"=== Creating Habitat-Sim Configuration ===")
        
        # Backend configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path
        backend_cfg.enable_physics = enable_physics
        backend_cfg.create_renderer = True
        backend_cfg.leave_context_with_background_renderer = False
        
        # Create default sensor specifications
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.uuid = "color_sensor"
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = [512, 512]
        camera_sensor_spec.position = [0.0, agent_height, 0.0]  # At agent's eye level
        camera_sensor_spec.orientation = [0.0, 0.0, 0.0]
        
        # Depth sensor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor" 
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [512, 512]
        depth_sensor_spec.position = [0.0, agent_height, 0.0]
        depth_sensor_spec.orientation = [0.0, 0.0, 0.0]
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [camera_sensor_spec, depth_sensor_spec]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec("move_forward", 
                habitat_sim.agent.ActuationSpec(amount=0.1)),
            "turn_left": habitat_sim.agent.ActionSpec("turn_left",
                habitat_sim.agent.ActuationSpec(amount=5.0)),  # degrees
            "turn_right": habitat_sim.agent.ActionSpec("turn_right", 
                habitat_sim.agent.ActuationSpec(amount=5.0)),
            "move_backward": habitat_sim.agent.ActionSpec("move_backward",
                habitat_sim.agent.ActuationSpec(amount=0.1)),
        }
        
        # Physics configuration - use default settings
        if enable_physics:
            # Physics will use default configuration when enable_physics=True
            pass
        
        # Full configuration
        configuration = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        print(f"Physics enabled: {enable_physics}")
        print(f"Agent radius: {agent_radius}")
        print(f"Agent height: {agent_height}")
        print(f"Sensors: RGB + Depth")
        print(f"Actions: move_forward, move_backward, turn_left, turn_right")
        
        return configuration
    
    def initialize_simulator(self, config):
        """
        Initialize the Habitat-Sim simulator
        """
        print(f"=== Initializing Habitat-Sim ===")
        
        try:
            # Close existing simulator if any
            if self.sim is not None:
                self.sim.close()
                
            # Create new simulator
            self.sim = habitat_sim.Simulator(config)
            
            print("Simulator initialized successfully!")
            
            # Print scene info (skip bounds for now - will get from pathfinder)
            scene_graph = self.sim.get_active_scene_graph()
            print(f"Scene graph initialized with root node")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize simulator: {e}")
            return False
    
    def setup_navigation(self, agent_radius=0.1, agent_height=1.5):
        """
        Setup navigation mesh for the scene
        """
        print(f"=== Setting Up Navigation ===")
        
        # Configure NavMesh settings
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.agent_radius = agent_radius
        navmesh_settings.agent_height = agent_height
        navmesh_settings.agent_max_climb = 0.2  # Max step height
        navmesh_settings.agent_max_slope = 45   # Max slope in degrees
        
        # Build NavMesh
        success = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        
        if success:
            print("NavMesh computed successfully!")
            
            # Get navigation info
            if self.sim.pathfinder.is_loaded:
                bounds = self.sim.pathfinder.get_bounds()
                print(f"NavMesh bounds: {bounds}")
                
                # Try to get a random navigable point
                nav_point = self.sim.pathfinder.get_random_navigable_point()
                print(f"Sample navigable point: {nav_point}")
                
                return nav_point
            else:
                print("Warning: NavMesh not loaded properly")
                return None
        else:
            print("Warning: Failed to compute NavMesh")
            return None
    
    def place_agent(self, position=None):
        """
        Place agent in the scene
        
        Args:
            position: [x, y, z] position. If None, uses random navigable point
        """
        print(f"=== Placing Agent ===")
        
        # Get a valid position
        if position is None:
            if self.sim.pathfinder.is_loaded:
                position = self.sim.pathfinder.get_random_navigable_point()
                print(f"Using random navigable point: {position}")
            else:
                # Fallback to origin with reasonable height
                position = [0.0, 1.0, 0.0]  # Simple fallback
                print(f"Using fallback position: {position}")
        
        # Create agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = position
        agent_state.rotation = np.quaternion(1, 0, 0, 0)  # Default orientation
        
        # Initialize agent
        agent = self.sim.initialize_agent(0, agent_state)
        
        print(f"Agent placed at: {agent_state.position}")
        print(f"Agent rotation: {agent_state.rotation}")
        
        return agent_state
    
    def visualize_scene(self, steps=100):
        """
        Visualize the scene with the agent
        """
        print(f"=== Visualizing Scene ===")
        
        if self.sim is None:
            print("Simulator not initialized!")
            return
        
        print("Taking sensor observations...")
        
        # Get initial observations
        observations = self.sim.get_sensor_observations()
        
        if "color_sensor" in observations:
            rgb = observations["color_sensor"]
            print(f"RGB observation shape: {rgb.shape}")
        
        if "depth_sensor" in observations:
            depth = observations["depth_sensor"]
            print(f"Depth observation shape: {depth.shape}")
        
        # Simple movement test
        print(f"Testing agent movement for {steps} steps...")
        
        for i in range(steps):
            # Random action
            action = np.random.choice(["move_forward", "turn_left", "turn_right", "move_backward"])
            
            # Take action
            try:
                observations = self.sim.step(action)
                
                # Print progress every 20 steps
                if i % 20 == 0:
                    agent_state = self.sim.get_agent(0).get_state()
                    print(f"Step {i}: Position {agent_state.position}")
                    
            except Exception as e:
                print(f"Action failed at step {i}: {e}")
                break
        
        print("Visualization complete!")
        
        # Final agent state
        final_state = self.sim.get_agent(0).get_state()
        print(f"Final agent position: {final_state.position}")
        
        return observations

    def show_agent_view(self, save_images=True, num_actions=10):
        """
        Show what the agent sees and allow manual control
        """
        import matplotlib.pyplot as plt
        
        print(f"=== Agent View Visualization ===")
        
        if self.sim is None:
            print("Simulator not initialized!")
            return
        
        # Get current observations
        observations = self.sim.get_sensor_observations()
        
        # Create figure for displaying observations
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        def update_display():
            # Get fresh observations
            obs = self.sim.get_sensor_observations()
            
            # Show RGB
            if "color_sensor" in obs:
                axes[0].clear()
                axes[0].imshow(obs["color_sensor"])
                axes[0].set_title("RGB View")
                axes[0].axis('off')
            
            # Show depth
            if "depth_sensor" in obs:
                axes[1].clear()
                depth_img = obs["depth_sensor"]
                # Normalize depth for display
                depth_display = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
                axes[1].imshow(depth_display, cmap='viridis')
                axes[1].set_title("Depth View")
                axes[1].axis('off')
            
            # Show agent position
            agent_state = self.sim.get_agent(0).get_state()
            fig.suptitle(f"Agent Position: [{agent_state.position[0]:.2f}, {agent_state.position[1]:.2f}, {agent_state.position[2]:.2f}]")
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
        
        # Initial display
        update_display()
        
        # Save initial view if requested
        if save_images:
            plt.savefig(os.path.join(self.output_dir, "agent_initial_view.png"), dpi=150, bbox_inches='tight')
            print(f"Saved initial view to: {self.output_dir}/agent_initial_view.png")
        
        # Demonstrate agent movement
        actions = ["move_forward", "turn_left", "turn_right", "move_backward"]
        
        print(f"\nDemonstrating {num_actions} random actions...")
        print("Actions: w=forward, a=turn_left, d=turn_right, s=backward")
        
        for i in range(num_actions):
            action = np.random.choice(actions)
            print(f"Step {i+1}: {action}")
            
            try:
                # Take action
                self.sim.step(action)
                
                # Update display
                update_display()
                
                # Save image if requested
                if save_images:
                    plt.savefig(os.path.join(self.output_dir, f"agent_view_step_{i+1:02d}.png"), 
                               dpi=150, bbox_inches='tight')
                
                # Small delay to see the movement
                plt.pause(0.5)
                
            except Exception as e:
                print(f"Action failed: {e}")
                break
        
        plt.show()
        
        if save_images:
            print(f"Agent view images saved to: {self.output_dir}/")
        
        return observations
    
    def interactive_control(self):
        """
        Interactive control of the agent (requires keyboard input)
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        
        print("=== Interactive Agent Control ===")
        print("Use the buttons to control the agent")
        
        if self.sim is None:
            print("Simulator not initialized!")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2)
        
        def update_display():
            obs = self.sim.get_sensor_observations()
            agent_state = self.sim.get_agent(0).get_state()
            
            # Clear all axes
            for ax in axes.flat:
                ax.clear()
            
            # RGB view
            if "color_sensor" in obs:
                axes[0,0].imshow(obs["color_sensor"])
                axes[0,0].set_title("RGB View")
                axes[0,0].axis('off')
            
            # Depth view
            if "depth_sensor" in obs:
                depth_img = obs["depth_sensor"]
                depth_display = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
                axes[0,1].imshow(depth_display, cmap='viridis')
                axes[0,1].set_title("Depth View")
                axes[0,1].axis('off')
            
            # Agent position plot
            pos = agent_state.position
            axes[1,0].scatter(pos[0], pos[2], c='red', s=100, marker='o')
            axes[1,0].set_title(f"Agent Position\n({pos[0]:.2f}, {pos[2]:.2f})")
            axes[1,0].grid(True)
            axes[1,0].set_aspect('equal')
            
            # Status
            axes[1,1].text(0.1, 0.7, f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", fontsize=12)
            axes[1,1].text(0.1, 0.5, f"Physics: {self.sim.config.sim_cfg.enable_physics}", fontsize=12)
            axes[1,1].text(0.1, 0.3, f"NavMesh: {self.sim.pathfinder.is_loaded}", fontsize=12)
            axes[1,1].set_title("Status")
            axes[1,1].axis('off')
            
            plt.draw()
        
        # Action functions
        def move_forward(event):
            self.sim.step("move_forward")
            update_display()
        
        def move_backward(event):
            self.sim.step("move_backward") 
            update_display()
        
        def turn_left(event):
            self.sim.step("turn_left")
            update_display()
        
        def turn_right(event):
            self.sim.step("turn_right")
            update_display()
        
        # Create buttons
        ax_forward = plt.axes([0.4, 0.05, 0.2, 0.04])
        ax_backward = plt.axes([0.4, 0.01, 0.2, 0.04])
        ax_left = plt.axes([0.2, 0.05, 0.2, 0.04])
        ax_right = plt.axes([0.6, 0.05, 0.2, 0.04])
        
        btn_forward = Button(ax_forward, 'Forward')
        btn_backward = Button(ax_backward, 'Backward')
        btn_left = Button(ax_left, 'Turn Left')
        btn_right = Button(ax_right, 'Turn Right')
        
        btn_forward.on_clicked(move_forward)
        btn_backward.on_clicked(move_backward)
        btn_left.on_clicked(turn_left)
        btn_right.on_clicked(turn_right)
        
        # Initial display
        update_display()
        
        plt.show()
        
        return None
    
    def get_scene_stats(self):
        """
        Get comprehensive scene statistics
        """
        if self.sim is None:
            return {}
            
        stats = {
            "physics_enabled": self.sim.config.sim_cfg.enable_physics,
            "navmesh_loaded": self.sim.pathfinder.is_loaded,
        }
        
        if self.sim.pathfinder.is_loaded:
            try:
                bounds = self.sim.pathfinder.get_bounds()
                stats["navmesh_bounds"] = f"min: {bounds[0]}, max: {bounds[1]}"
            except:
                stats["navmesh_bounds"] = "unavailable"
                
            try:
                stats["navigable_area"] = self.sim.pathfinder.navigable_area
            except:
                stats["navigable_area"] = "unknown"
        
        return stats
    
    def cleanup(self):
        """
        Clean up simulator resources
        """
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            print("Simulator closed")


def setup_habitat_environment(rl_env, stats):
    """
    Complete setup pipeline from RL environment to Habitat-Sim
    
    Args:
        rl_env: Output from create_and_preview_rl_environment()
        stats: Stats from create_and_preview_rl_environment()
    """
    print("=" * 60)
    print("HABITAT-SIM ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Initialize setup
    habitat_setup = HabitatEnvSetup(rl_env)
    
    try:
        # Step 1: Export scene mesh
        scene_path = habitat_setup.export_scene_mesh(format="ply")
        
        # Step 2: Create configuration
        config = habitat_setup.create_habitat_config(
            enable_physics=True,
            agent_radius=0.15,  # Slightly larger for stability
            agent_height=1.5
        )
        
        # Step 3: Initialize simulator
        success = habitat_setup.initialize_simulator(config)
        if not success:
            return None
        
        # Step 4: Setup navigation
        nav_point = habitat_setup.setup_navigation(
            agent_radius=0.15,
            agent_height=1.5
        )
        
        # Step 5: Place agent
        agent_state = habitat_setup.place_agent()
        
        # Step 6: Get scene statistics
        scene_stats = habitat_setup.get_scene_stats()
        print("\n=== Scene Statistics ===")
        for key, value in scene_stats.items():
            print(f"{key}: {value}")
        
        # Step 7: Visualize (brief test)
        print("\n=== Running Visualization Test ===")
        observations = habitat_setup.visualize_scene(steps=50)
        
        print("\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        print(f"Scene file: {scene_path}")
        print(f"Physics: Enabled")
        print(f"Navigation: {'Available' if nav_point is not None else 'Limited'}")
        print(f"Agent: Placed and functional")
        print("\nReady for RL training!")
        
        return habitat_setup
        
    except Exception as e:
        print(f"Setup failed: {e}")
        habitat_setup.cleanup()
        return None

    
# Main execution
recon_path = "reconstruction/sparse/0"

detector = CameraViewGroundDetector(recon_path)
results = detector.detect_ground_and_normalize()

np.savez('camera_view_ground_env.npz',
    points=results['normalized_data']['points'],
    colors=results['normalized_data']['colors'],
    camera_positions=results['normalized_data']['camera_positions'],
    camera_directions=results['normalized_data']['camera_directions'])

# Load and create RL environment
data = np.load('camera_view_ground_env.npz')
normalized_data = {
    'points': data['points'],
    'colors': data['colors'],
    'camera_positions': data['camera_positions'],
    'camera_directions': data['camera_directions']
}
rl_env, stats = create_and_preview_rl_environment(normalized_data, add_walls=True, visualize=False)
habitat_setup = setup_habitat_environment(rl_env, stats)
# habitat_setup.show_agent_view(save_images=True, num_actions=15)
scene_path = habitat_setup.scene_path
try:
    # This opens an interactive 3D viewer
    subprocess.run([sys.executable, "-m", "habitat_sim.utils.viewer", scene_path])
except Exception as e:
    print(f"Viewer failed: {e}")